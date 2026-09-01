#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(__unix__) && !defined(__APPLE__)
#error "The N-body browser viewer server requires POSIX sockets"
#endif

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "streaming/frame.hpp"

#ifndef NBODY_BROWSER_ASSET_DIR
#define NBODY_BROWSER_ASSET_DIR "apps/browser_viewer/web"
#endif

namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

volatile std::sig_atomic_t stopRequested = 0;
void requestStop(int) { stopRequested = 1; }

struct ViewerConfig {
  fs::path framesDirectory;
  fs::path assetsDirectory = NBODY_BROWSER_ASSET_DIR;
  std::string bindAddress = "0.0.0.0";
  std::uint16_t port = 8080;
  std::chrono::milliseconds pollInterval{500};
};

struct FrameInfo {
  fs::path path;
  std::string name;
  std::string sessionId;
  nbody::streaming::FrameHeader header;
  fs::file_time_type modified{};
};

struct CatalogSnapshot {
  std::string sessionId;
  bool complete = false;
  std::vector<FrameInfo> frames;
};

class Socket {
public:
  Socket() = default;
  explicit Socket(int descriptor) : descriptor_(descriptor) {}
  ~Socket() {
    if (descriptor_ >= 0)
      ::close(descriptor_);
  }

  Socket(const Socket &) = delete;
  Socket &operator=(const Socket &) = delete;
  Socket(Socket &&other) noexcept : descriptor_(other.descriptor_) {
    other.descriptor_ = -1;
  }
  Socket &operator=(Socket &&other) noexcept {
    if (this != &other) {
      if (descriptor_ >= 0)
        ::close(descriptor_);
      descriptor_ = other.descriptor_;
      other.descriptor_ = -1;
    }
    return *this;
  }

  [[nodiscard]] int get() const noexcept { return descriptor_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return descriptor_ >= 0;
  }

private:
  int descriptor_ = -1;
};

std::string socketError(std::string_view operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

std::string_view requireValue(int argc, char **argv, int &index,
                              std::string_view option) {
  if (index + 1 >= argc)
    throw std::invalid_argument("Missing value for " + std::string(option));
  return argv[++index];
}

unsigned long parsePositive(std::string_view text, std::string_view option,
                            unsigned long maximum) {
  unsigned long value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value == 0 || value > maximum) {
    throw std::invalid_argument("Invalid value for " + std::string(option) +
                                ": " + std::string(text));
  }
  return value;
}

void printUsage(std::ostream &output, const char *program) {
  output
      << "Usage: " << program << " --frames-dir DIR [OPTIONS]\n\n"
      << "  --frames-dir DIR  Directory containing completed .nbsnap files\n"
      << "  --bind ADDRESS    HTTP bind address (default: 0.0.0.0)\n"
      << "  --port PORT       HTTP port (default: 8080)\n"
      << "  --poll-ms MS      Directory refresh interval (default: 500)\n"
      << "  --assets-dir DIR  Browser asset directory\n"
      << "  -h, --help        Show this help and exit\n";
}

ViewerConfig parseCommandLine(int argc, char **argv, bool &showHelp) {
  ViewerConfig config;
  showHelp = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "-h" || option == "--help") {
      showHelp = true;
      return config;
    }
    if (option == "--frames-dir")
      config.framesDirectory = requireValue(argc, argv, index, option);
    else if (option == "--bind")
      config.bindAddress = requireValue(argc, argv, index, option);
    else if (option == "--port")
      config.port = static_cast<std::uint16_t>(parsePositive(
          requireValue(argc, argv, index, option), option, 65535));
    else if (option == "--poll-ms")
      config.pollInterval = std::chrono::milliseconds(parsePositive(
          requireValue(argc, argv, index, option), option, 60000));
    else if (option == "--assets-dir")
      config.assetsDirectory = requireValue(argc, argv, index, option);
    else
      throw std::invalid_argument("Unknown option: " + std::string(option));
  }

  if (config.framesDirectory.empty())
    throw std::invalid_argument("--frames-dir is required");
  if (config.bindAddress.empty())
    throw std::invalid_argument("--bind must not be empty");
  if (config.assetsDirectory.empty())
    throw std::invalid_argument("--assets-dir must not be empty");
  return config;
}

std::optional<std::string> sessionIdFromFrameName(std::string_view name) {
  constexpr std::string_view prefix = "run-";
  constexpr std::string_view separator = "-frame-";
  constexpr std::string_view suffix = ".nbsnap";
  if (!name.starts_with(prefix) || !name.ends_with(suffix))
    return std::nullopt;
  const std::size_t separatorPosition = name.rfind(separator);
  if (separatorPosition == std::string_view::npos ||
      separatorPosition <= prefix.size())
    return std::nullopt;
  return std::string(
      name.substr(prefix.size(), separatorPosition - prefix.size()));
}

std::string completionName(std::string_view sessionId) {
  return "run-" + std::string(sessionId) + ".complete";
}

class FrameCatalog {
public:
  FrameCatalog(fs::path directory, std::chrono::milliseconds refreshInterval)
      : directory_(std::move(directory)), refreshInterval_(refreshInterval) {}

  CatalogSnapshot snapshot() {
    std::scoped_lock lock(mutex_);
    refreshIfNeeded();
    return snapshot_;
  }

  std::optional<FrameInfo> findFrame(std::string_view name) {
    std::scoped_lock lock(mutex_);
    refreshIfNeeded();
    const auto found = std::find_if(
        snapshot_.frames.begin(), snapshot_.frames.end(),
        [&](const FrameInfo &frame) { return frame.name == name; });
    if (found == snapshot_.frames.end())
      return std::nullopt;
    return *found;
  }

private:
  void refreshIfNeeded() {
    const auto now = Clock::now();
    if (initialized_ && now - lastRefresh_ < refreshInterval_)
      return;
    lastRefresh_ = now;

    std::error_code error;
    const auto directoryModified = fs::last_write_time(directory_, error);
    if (error)
      throw std::runtime_error("Could not inspect frame directory '" +
                               directory_.string() + "': " + error.message());
    if (initialized_ && directoryModified == directoryModified_)
      return;
    directoryModified_ = directoryModified;

    std::map<std::string, std::vector<FrameInfo>> runs;
    fs::directory_iterator entries(directory_, error);
    if (error)
      throw std::runtime_error("Could not scan frame directory '" +
                               directory_.string() + "': " + error.message());

    for (const auto &entry : entries) {
      error.clear();
      if (!entry.is_regular_file(error) || error ||
          entry.path().extension() != ".nbsnap")
        continue;
      const std::string name = entry.path().filename().string();
      const auto session = sessionIdFromFrameName(name);
      if (!session)
        continue;
      try {
        FrameInfo frame;
        frame.path = entry.path();
        frame.name = name;
        frame.sessionId = *session;
        frame.header = nbody::streaming::readFrameHeader(entry.path());
        frame.modified = entry.last_write_time(error);
        if (!error)
          runs[frame.sessionId].push_back(std::move(frame));
      } catch (const std::exception &validationError) {
        std::cerr << "Ignoring invalid frame '" << entry.path()
                  << "': " << validationError.what() << '\n';
      }
    }

    CatalogSnapshot next;
    fs::file_time_type newestRunTime = fs::file_time_type::min();
    for (auto &[session, frames] : runs) {
      if (frames.empty())
        continue;
      const auto newest = std::max_element(
          frames.begin(), frames.end(), [](const FrameInfo &left,
                                           const FrameInfo &right) {
            return left.modified < right.modified;
          });
      if (next.sessionId.empty() || newest->modified > newestRunTime) {
        newestRunTime = newest->modified;
        next.sessionId = session;
        next.frames = std::move(frames);
      }
    }

    std::sort(next.frames.begin(), next.frames.end(),
              [](const FrameInfo &left, const FrameInfo &right) {
                if (left.header.sequence != right.header.sequence)
                  return left.header.sequence < right.header.sequence;
                return left.name < right.name;
              });
    if (!next.sessionId.empty()) {
      error.clear();
      next.complete = fs::is_regular_file(
          directory_ / completionName(next.sessionId), error);
      if (error)
        next.complete = false;
    }
    snapshot_ = std::move(next);
    initialized_ = true;
  }

  fs::path directory_;
  std::chrono::milliseconds refreshInterval_;
  Clock::time_point lastRefresh_{};
  fs::file_time_type directoryModified_{};
  CatalogSnapshot snapshot_;
  std::mutex mutex_;
  bool initialized_ = false;
};

std::string jsonEscape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size() + 8);
  for (const unsigned char character : text) {
    switch (character) {
    case '"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\b':
      escaped += "\\b";
      break;
    case '\f':
      escaped += "\\f";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (character < 0x20) {
        constexpr char digits[] = "0123456789abcdef";
        escaped += "\\u00";
        escaped += digits[(character >> 4U) & 0x0fU];
        escaped += digits[character & 0x0fU];
      } else {
        escaped.push_back(static_cast<char>(character));
      }
    }
  }
  return escaped;
}

std::string catalogJson(const CatalogSnapshot &snapshot) {
  std::ostringstream output;
  output.precision(17);
  output << "{\"runId\":\"" << jsonEscape(snapshot.sessionId)
         << "\",\"complete\":" << (snapshot.complete ? "true" : "false")
         << ",\"frameCount\":" << snapshot.frames.size()
         << ",\"frames\":[";
  for (std::size_t index = 0; index < snapshot.frames.size(); ++index) {
    if (index > 0)
      output << ',';
    const FrameInfo &frame = snapshot.frames[index];
    output << "{\"name\":\"" << jsonEscape(frame.name)
           << "\",\"sequence\":\"" << frame.header.sequence
           << "\",\"step\":\"" << frame.header.simulationStep
           << "\",\"time\":" << frame.header.simulationTime
           << ",\"particles\":" << frame.header.particleCount
           << ",\"sourceParticles\":" << frame.header.sourceParticleCount
           << ",\"version\":" << frame.header.version
           << ",\"hasTypes\":"
           << (frame.header.particleTypeBytes == frame.header.particleCount
                   ? "true"
                   : "false")
           << '}';
  }
  output << "]}";
  return output.str();
}

void configureSocket(int descriptor) {
  timeval timeout{};
  timeout.tv_sec = 10;
  (void)::setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                     sizeof(timeout));
  (void)::setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                     sizeof(timeout));
}

Socket createListener(const std::string &bindAddress, std::uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  addrinfo *addresses = nullptr;
  const std::string service = std::to_string(port);
  const int status = ::getaddrinfo(bindAddress.c_str(), service.c_str(), &hints,
                                   &addresses);
  if (status != 0)
    throw std::runtime_error("Could not resolve bind address '" + bindAddress +
                             "': " + gai_strerror(status));

  Socket listener;
  std::string lastError = "no usable address";
  for (addrinfo *address = addresses; address != nullptr;
       address = address->ai_next) {
    Socket candidate(
        ::socket(address->ai_family, address->ai_socktype, address->ai_protocol));
    if (!candidate) {
      lastError = socketError("socket");
      continue;
    }
    const int enabled = 1;
    (void)::setsockopt(candidate.get(), SOL_SOCKET, SO_REUSEADDR, &enabled,
                       sizeof(enabled));
    if (::bind(candidate.get(), address->ai_addr, address->ai_addrlen) != 0) {
      lastError = socketError("bind");
      continue;
    }
    if (::listen(candidate.get(), 32) != 0) {
      lastError = socketError("listen");
      continue;
    }
    listener = std::move(candidate);
    break;
  }
  ::freeaddrinfo(addresses);
  if (!listener)
    throw std::runtime_error("Could not listen on " + bindAddress + ':' +
                             service + " (" + lastError + ')');
  return listener;
}

void sendAll(int descriptor, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const char *>(data);
  while (bytes > 0) {
#if defined(MSG_NOSIGNAL)
    const ssize_t sent = ::send(descriptor, cursor, bytes, MSG_NOSIGNAL);
#else
    const ssize_t sent = ::send(descriptor, cursor, bytes, 0);
#endif
    if (sent < 0 && errno == EINTR)
      continue;
    if (sent <= 0)
      throw std::runtime_error(socketError("HTTP send failed"));
    cursor += sent;
    bytes -= static_cast<std::size_t>(sent);
  }
}

void sendHeader(int descriptor, int status, std::string_view statusText,
                std::string_view contentType, std::uintmax_t contentBytes,
                std::string_view cacheControl = "no-store") {
  std::ostringstream header;
  header << "HTTP/1.1 " << status << ' ' << statusText << "\r\n"
         << "Content-Type: " << contentType << "\r\n"
         << "Content-Length: " << contentBytes << "\r\n"
         << "Cache-Control: " << cacheControl << "\r\n"
         << "X-Content-Type-Options: nosniff\r\n"
         << "Referrer-Policy: no-referrer\r\n"
         << "Cross-Origin-Resource-Policy: same-origin\r\n"
         << "Content-Security-Policy: default-src 'self'; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; img-src 'self' data:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'\r\n"
         << "Connection: close\r\n\r\n";
  const std::string text = header.str();
  sendAll(descriptor, text.data(), text.size());
}

void sendText(int descriptor, int status, std::string_view statusText,
              std::string_view contentType, const std::string &body,
              bool headOnly = false) {
  sendHeader(descriptor, status, statusText, contentType, body.size());
  if (!headOnly && !body.empty())
    sendAll(descriptor, body.data(), body.size());
}

void sendError(int descriptor, int status, std::string_view statusText,
               std::string_view message, bool headOnly = false) {
  const std::string body = "{\"error\":\"" + jsonEscape(message) + "\"}";
  sendText(descriptor, status, statusText, "application/json; charset=utf-8",
           body, headOnly);
}

void sendFile(int descriptor, const fs::path &path,
              std::string_view contentType, bool headOnly,
              std::string_view cacheControl) {
  std::error_code error;
  const auto bytes = fs::file_size(path, error);
  if (error)
    throw std::runtime_error("Could not determine file size: " +
                             error.message());
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open file: " + path.string());
  sendHeader(descriptor, 200, "OK", contentType, bytes, cacheControl);
  if (headOnly)
    return;
  std::array<char, 256 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize read = input.gcount();
    if (read > 0)
      sendAll(descriptor, buffer.data(), static_cast<std::size_t>(read));
  }
}

struct HttpRequest {
  std::string method;
  std::string target;
};

HttpRequest readRequest(int descriptor) {
  std::string request;
  request.reserve(2048);
  std::array<char, 2048> buffer{};
  while (request.find("\r\n\r\n") == std::string::npos) {
    const ssize_t received = ::recv(descriptor, buffer.data(), buffer.size(), 0);
    if (received < 0 && errno == EINTR)
      continue;
    if (received <= 0)
      throw std::runtime_error("Client disconnected before sending a request");
    request.append(buffer.data(), static_cast<std::size_t>(received));
    if (request.size() > 16 * 1024)
      throw std::runtime_error("HTTP request header is too large");
  }

  const std::size_t lineEnd = request.find("\r\n");
  std::istringstream line(request.substr(0, lineEnd));
  HttpRequest parsed;
  std::string version;
  line >> parsed.method >> parsed.target >> version;
  if (!line || (version != "HTTP/1.1" && version != "HTTP/1.0"))
    throw std::runtime_error("Malformed HTTP request line");
  const std::size_t query = parsed.target.find('?');
  if (query != std::string::npos)
    parsed.target.resize(query);
  return parsed;
}

void serveRequest(int descriptor, const ViewerConfig &config,
                  FrameCatalog &catalog) {
  const HttpRequest request = readRequest(descriptor);
  const bool headOnly = request.method == "HEAD";
  if (request.method != "GET" && !headOnly) {
    sendError(descriptor, 405, "Method Not Allowed", "Only GET is supported");
    return;
  }

  if (request.target == "/api/health") {
    sendText(descriptor, 200, "OK", "application/json; charset=utf-8",
             "{\"status\":\"ok\"}", headOnly);
    return;
  }
  if (request.target == "/api/frames") {
    sendText(descriptor, 200, "OK", "application/json; charset=utf-8",
             catalogJson(catalog.snapshot()), headOnly);
    return;
  }

  constexpr std::string_view framePrefix = "/api/frame/";
  if (request.target.starts_with(framePrefix)) {
    const std::string name(request.target.substr(framePrefix.size()));
    if (name.empty() || name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos || name.find("..") != std::string::npos) {
      sendError(descriptor, 400, "Bad Request", "Invalid frame name", headOnly);
      return;
    }
    const auto frame = catalog.findFrame(name);
    if (!frame) {
      sendError(descriptor, 404, "Not Found", "Frame not found", headOnly);
      return;
    }
    sendFile(descriptor, frame->path, "application/octet-stream", headOnly,
             "public, max-age=31536000, immutable");
    return;
  }

  const std::map<std::string, std::pair<std::string, std::string>> assets{
      {"/", {"index.html", "text/html; charset=utf-8"}},
      {"/index.html", {"index.html", "text/html; charset=utf-8"}},
      {"/styles.css", {"styles.css", "text/css; charset=utf-8"}},
      {"/app.js", {"app.js", "text/javascript; charset=utf-8"}},
      {"/frame_parser.mjs",
       {"frame_parser.mjs", "text/javascript; charset=utf-8"}},
  };
  const auto asset = assets.find(request.target);
  if (asset == assets.end()) {
    sendError(descriptor, 404, "Not Found", "Resource not found", headOnly);
    return;
  }
  sendFile(descriptor, config.assetsDirectory / asset->second.first,
           asset->second.second, headOnly, "no-cache");
}

bool isWildcardAddress(std::string_view address) {
  return address == "0.0.0.0" || address == "::" || address == "*";
}

std::vector<std::string> discoverAccessAddresses(std::string_view bindAddress) {
  if (!isWildcardAddress(bindAddress))
    return {std::string(bindAddress)};

  std::vector<std::string> addresses;
  ifaddrs *interfaces = nullptr;
  if (::getifaddrs(&interfaces) != 0)
    return {"127.0.0.1"};
  for (ifaddrs *interface = interfaces; interface != nullptr;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == nullptr ||
        (interface->ifa_flags & IFF_UP) == 0 ||
        (interface->ifa_flags & IFF_LOOPBACK) != 0)
      continue;
    const int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;
    char host[NI_MAXHOST]{};
    const socklen_t length = family == AF_INET ? sizeof(sockaddr_in)
                                               : sizeof(sockaddr_in6);
    if (::getnameinfo(interface->ifa_addr, length, host, sizeof(host), nullptr,
                      0, NI_NUMERICHOST) != 0)
      continue;
    std::string address(host);
    if (address.find('%') != std::string::npos)
      continue;
    if (std::find(addresses.begin(), addresses.end(), address) ==
        addresses.end())
      addresses.push_back(std::move(address));
  }
  ::freeifaddrs(interfaces);
  if (addresses.empty())
    addresses.push_back("127.0.0.1");
  return addresses;
}

std::string httpUrl(std::string_view address, std::uint16_t port) {
  const bool ipv6 = address.find(':') != std::string_view::npos;
  return "http://" + std::string(ipv6 ? "[" : "") + std::string(address) +
         (ipv6 ? "]" : "") + ':' + std::to_string(port) + '/';
}

void validateConfig(const ViewerConfig &config) {
  std::error_code error;
  if (!fs::is_directory(config.framesDirectory, error) || error)
    throw std::invalid_argument("Frame directory does not exist or is not a "
                                "directory: " +
                                config.framesDirectory.string());
  for (const std::string_view asset :
       {"index.html", "styles.css", "app.js", "frame_parser.mjs"}) {
    error.clear();
    if (!fs::is_regular_file(config.assetsDirectory / asset, error) || error)
      throw std::invalid_argument("Browser asset is missing: " +
                                  (config.assetsDirectory / asset).string());
  }
}

void runServer(const ViewerConfig &config) {
  validateConfig(config);
  FrameCatalog catalog(config.framesDirectory, config.pollInterval);
  Socket listener = createListener(config.bindAddress, config.port);

  const auto addresses = discoverAccessAddresses(config.bindAddress);
  std::cout << "N-body browser viewer is ready\n"
            << "  frames: " << fs::absolute(config.framesDirectory) << '\n'
            << "  access: " << httpUrl(addresses.front(), config.port) << '\n';
  for (std::size_t index = 1; index < addresses.size(); ++index)
    std::cout << "  access: " << httpUrl(addresses[index], config.port) << '\n';
  if (isWildcardAddress(config.bindAddress)) {
    std::cout << "  note: the viewer has no built-in authentication; use the "
                 "cluster firewall or an SSH tunnel\n";
  }
  std::cout << std::flush;

  while (!stopRequested) {
    sockaddr_storage peer{};
    socklen_t peerBytes = sizeof(peer);
    const int descriptor =
        ::accept(listener.get(), reinterpret_cast<sockaddr *>(&peer), &peerBytes);
    if (descriptor < 0 && errno == EINTR)
      continue;
    if (descriptor < 0)
      throw std::runtime_error(socketError("Could not accept HTTP client"));
    try {
      Socket client(descriptor);
      configureSocket(client.get());
      serveRequest(client.get(), config, catalog);
    } catch (const std::exception &error) {
      std::cerr << "Viewer request failed: " << error.what() << '\n';
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    bool showHelp = false;
    const ViewerConfig config = parseCommandLine(argc, argv, showHelp);
    if (showHelp) {
      printUsage(std::cout, argv[0]);
      return 0;
    }
    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    runServer(config);
    std::cout << "N-body browser viewer stopped\n";
    return 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Browser viewer failed: " << error.what() << '\n';
    return 1;
  }
}
