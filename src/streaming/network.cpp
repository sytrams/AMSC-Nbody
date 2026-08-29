#include "streaming/network.hpp"

#include "streaming/frame.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if !defined(__unix__) && !defined(__APPLE__)
#error "N-body streaming currently requires POSIX sockets"
#endif

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

namespace nbody::streaming {
namespace {

constexpr std::array<char, 8> kProtocolMagic{'N', 'B', 'S', 'T',
                                             'R', 'M', '0', '1'};
constexpr std::array<char, 8> kRelayMagic{'N', 'B', 'R', 'E',
                                          'L', 'A', 'Y', '1'};
constexpr std::uint32_t kFrameMessage = 1;
constexpr std::uint32_t kEndMessage = 2;
constexpr std::uint32_t kAckMessage = 1;
constexpr std::size_t kIoBufferBytes = 256 * 1024;
constexpr std::size_t kMinimumRelayTokenBytes = 32;
constexpr std::size_t kMaximumRelayTokenBytes = 4096;
constexpr std::uint8_t kRelayRejected = 0;
constexpr std::uint8_t kRelayAccepted = 1;
constexpr std::uint8_t kRelayGo = 2;

bool waitForReadable(int descriptor, std::chrono::milliseconds timeout);

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

std::string socketError(const std::string &operation) {
  return operation + ": " + std::strerror(errno);
}

void configureConnectedSocket(int descriptor) {
  timeval timeout{};
  timeout.tv_sec = 2;
  if (::setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout)) != 0 ||
      ::setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                   sizeof(timeout)) != 0) {
    throw std::runtime_error(socketError("Could not configure socket timeout"));
  }
}

Socket createListener(const std::string &bindAddress, std::uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  addrinfo *addresses = nullptr;
  const std::string service = std::to_string(port);
  const char *node = bindAddress.empty() ? nullptr : bindAddress.c_str();
  const int status = ::getaddrinfo(node, service.c_str(), &hints, &addresses);
  if (status != 0)
    throw std::runtime_error("Could not resolve bind address '" + bindAddress +
                             "': " + gai_strerror(status));

  Socket listener;
  std::string lastError = "no usable address";
  for (addrinfo *address = addresses; address != nullptr;
       address = address->ai_next) {
    Socket candidate(::socket(address->ai_family, address->ai_socktype,
                              address->ai_protocol));
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
    if (::listen(candidate.get(), 4) != 0) {
      lastError = socketError("listen");
      continue;
    }
    listener = std::move(candidate);
    break;
  }
  ::freeaddrinfo(addresses);

  if (!listener)
    throw std::runtime_error("Could not listen on " + bindAddress + ":" +
                             service + " (" + lastError + ")");
  return listener;
}

Socket connectTo(const std::string &host, std::uint16_t port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  addrinfo *addresses = nullptr;
  const std::string service = std::to_string(port);
  const int status =
      ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
  if (status != 0)
    throw std::runtime_error("Could not resolve server '" + host +
                             "': " + gai_strerror(status));

  Socket connection;
  std::string lastError = "no usable address";
  for (addrinfo *address = addresses; address != nullptr;
       address = address->ai_next) {
    Socket candidate(::socket(address->ai_family, address->ai_socktype,
                              address->ai_protocol));
    if (!candidate) {
      lastError = socketError("socket");
      continue;
    }
    if (::connect(candidate.get(), address->ai_addr, address->ai_addrlen) !=
        0) {
      lastError = socketError("connect");
      continue;
    }
    configureConnectedSocket(candidate.get());
    connection = std::move(candidate);
    break;
  }
  ::freeaddrinfo(addresses);

  if (!connection)
    throw std::runtime_error("Could not connect to " + host + ":" + service +
                             " (" + lastError + ")");
  return connection;
}

Socket acceptConnection(int listener) {
  sockaddr_storage peer{};
  socklen_t peerBytes = sizeof(peer);
  const int descriptor =
      ::accept(listener, reinterpret_cast<sockaddr *>(&peer), &peerBytes);
  if (descriptor < 0)
    throw std::runtime_error(socketError("Could not accept connection"));
  return Socket(descriptor);
}

void sendAll(int socket, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const char *>(data);
  while (bytes > 0) {
#if defined(MSG_NOSIGNAL)
    const ssize_t sent = ::send(socket, cursor, bytes, MSG_NOSIGNAL);
#else
    const ssize_t sent = ::send(socket, cursor, bytes, 0);
#endif
    if (sent < 0 && errno == EINTR)
      continue;
    if (sent <= 0)
      throw std::runtime_error(socketError("Socket send failed"));
    cursor += sent;
    bytes -= static_cast<std::size_t>(sent);
  }
}

void receiveAll(int socket, void *data, std::size_t bytes) {
  auto *cursor = static_cast<char *>(data);
  while (bytes > 0) {
    const ssize_t received = ::recv(socket, cursor, bytes, 0);
    if (received < 0 && errno == EINTR)
      continue;
    if (received == 0)
      throw std::runtime_error("Peer disconnected");
    if (received < 0)
      throw std::runtime_error(socketError("Socket receive failed"));
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
}

void sendU32(int socket, std::uint32_t value) {
  const std::uint32_t networkValue = htonl(value);
  sendAll(socket, &networkValue, sizeof(networkValue));
}

std::uint32_t receiveU32(int socket) {
  std::uint32_t networkValue = 0;
  receiveAll(socket, &networkValue, sizeof(networkValue));
  return ntohl(networkValue);
}

void sendU64(int socket, std::uint64_t value) {
  std::array<unsigned char, 8> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<unsigned char>(value >> ((7U - index) * 8U));
  sendAll(socket, bytes.data(), bytes.size());
}

std::uint64_t receiveU64(int socket) {
  std::array<unsigned char, 8> bytes{};
  receiveAll(socket, bytes.data(), bytes.size());
  std::uint64_t value = 0;
  for (const auto byte : bytes)
    value = (value << 8U) | byte;
  return value;
}

std::string readRelayToken(const std::filesystem::path &path) {
  if (path.empty())
    throw std::invalid_argument("Relay token file path must not be empty");

  struct stat tokenStatus {};
  if (::stat(path.c_str(), &tokenStatus) != 0)
    throw std::runtime_error("Could not inspect relay token file: " +
                             path.string());
  if (!S_ISREG(tokenStatus.st_mode))
    throw std::runtime_error("Relay token path is not a regular file: " +
                             path.string());
  if ((tokenStatus.st_mode & (S_IRWXG | S_IRWXO)) != 0)
    throw std::runtime_error(
        "Relay token file must not be accessible by group or other users: " +
        path.string());

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open relay token file: " +
                             path.string());

  std::string token((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  while (!token.empty() && (token.back() == '\n' || token.back() == '\r'))
    token.pop_back();

  if (token.size() < kMinimumRelayTokenBytes ||
      token.size() > kMaximumRelayTokenBytes)
    throw std::runtime_error(
        "Relay token must contain between 32 and 4096 bytes");
  if (std::any_of(token.begin(), token.end(), [](unsigned char byte) {
        return std::isspace(byte) != 0 || byte == 0;
      }))
    throw std::runtime_error(
        "Relay token must not contain whitespace or NUL bytes");
  return token;
}

bool constantTimeEqual(const std::string &left, const std::string &right) {
  const std::size_t compared = std::max(left.size(), right.size());
  unsigned int difference =
      static_cast<unsigned int>(left.size() ^ right.size());
  for (std::size_t index = 0; index < compared; ++index) {
    const unsigned char leftByte =
        index < left.size() ? static_cast<unsigned char>(left[index]) : 0U;
    const unsigned char rightByte =
        index < right.size() ? static_cast<unsigned char>(right[index]) : 0U;
    difference |= static_cast<unsigned int>(leftByte ^ rightByte);
  }
  return difference == 0U;
}

void sendRelayAuthentication(int socket, const std::string &token) {
  sendAll(socket, kRelayMagic.data(), kRelayMagic.size());
  sendU32(socket, static_cast<std::uint32_t>(token.size()));
  sendAll(socket, token.data(), token.size());
}

bool receiveRelayAuthentication(int socket, const std::string &expectedToken) {
  std::array<char, kRelayMagic.size()> magic{};
  receiveAll(socket, magic.data(), magic.size());
  if (magic != kRelayMagic)
    return false;

  const std::uint32_t tokenBytes = receiveU32(socket);
  if (tokenBytes < kMinimumRelayTokenBytes ||
      tokenBytes > kMaximumRelayTokenBytes)
    return false;

  std::string token(tokenBytes, '\0');
  receiveAll(socket, token.data(), token.size());
  return constantTimeEqual(token, expectedToken);
}

void waitForRelayGo(int socket, const std::function<bool()> &stopRequested) {
  while (!stopRequested()) {
    if (!waitForReadable(socket, std::chrono::milliseconds(250)))
      continue;
    std::uint8_t command = 0;
    receiveAll(socket, &command, sizeof(command));
    if (command != kRelayGo)
      throw std::runtime_error("Relay sent an invalid pairing command");
    return;
  }
}

Socket waitForRelayClient(int listener, int source,
                          const std::function<bool()> &stopRequested) {
  while (!stopRequested()) {
    std::array<pollfd, 2> descriptors{
        {{listener, POLLIN, 0}, {source, POLLIN, 0}}};
    const int result = ::poll(descriptors.data(), descriptors.size(), 250);
    if (result < 0 && errno == EINTR)
      continue;
    if (result < 0)
      throw std::runtime_error(socketError("Relay poll failed"));
    if (result == 0)
      continue;

    if ((descriptors[1].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0)
      throw std::runtime_error("Relay source disconnected while waiting");
    if ((descriptors[0].revents & POLLIN) != 0)
      return acceptConnection(listener);
    if ((descriptors[0].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
      throw std::runtime_error("Relay client listener failed");
  }
  return Socket();
}

void bridgeConnections(int source, int client,
                       const std::function<bool()> &stopRequested) {
  std::vector<char> buffer(kIoBufferBytes);
  while (!stopRequested()) {
    std::array<pollfd, 2> descriptors{
        {{source, POLLIN, 0}, {client, POLLIN, 0}}};
    const int result = ::poll(descriptors.data(), descriptors.size(), 250);
    if (result < 0 && errno == EINTR)
      continue;
    if (result < 0)
      throw std::runtime_error(socketError("Relay bridge poll failed"));
    if (result == 0)
      continue;

    for (std::size_t index = 0; index < descriptors.size(); ++index) {
      const short events = descriptors[index].revents;
      if ((events & POLLIN) != 0) {
        const int from = descriptors[index].fd;
        const int to = descriptors[1U - index].fd;
        const ssize_t received = ::recv(from, buffer.data(), buffer.size(), 0);
        if (received < 0 && errno == EINTR)
          continue;
        if (received <= 0)
          return;
        sendAll(to, buffer.data(), static_cast<std::size_t>(received));
      }
      // POLLIN and POLLHUP can arrive together when the peer closed after its
      // final message. Drain the readable bytes above before ending the pair.
      if ((events & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        return;
    }
  }
}

void interruptibleDelay(std::chrono::milliseconds delay,
                        const std::function<bool()> &stopRequested) {
  constexpr auto slice = std::chrono::milliseconds(50);
  auto remaining = delay;
  while (remaining.count() > 0 && !stopRequested()) {
    const auto current = std::min(remaining, slice);
    std::this_thread::sleep_for(current);
    remaining -= current;
  }
}

void sendHello(int socket, bool follow) {
  sendAll(socket, kProtocolMagic.data(), kProtocolMagic.size());
  const std::uint8_t mode = follow ? 1 : 0;
  sendAll(socket, &mode, sizeof(mode));
}

bool receiveHello(int socket) {
  std::array<char, kProtocolMagic.size()> magic{};
  receiveAll(socket, magic.data(), magic.size());
  if (magic != kProtocolMagic)
    throw std::runtime_error("Client used an unsupported streaming protocol");
  std::uint8_t mode = 0;
  receiveAll(socket, &mode, sizeof(mode));
  if (mode > 1)
    throw std::runtime_error("Client requested an invalid streaming mode");
  return mode == 1;
}

void sendMessageHeader(int socket, std::uint32_t type, const std::string &name,
                       std::uint64_t contentBytes) {
  if (name.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("Streaming file name is too long");
  sendU32(socket, type);
  sendU32(socket, static_cast<std::uint32_t>(name.size()));
  sendU64(socket, contentBytes);
  if (!name.empty())
    sendAll(socket, name.data(), name.size());
}

struct MessageHeader {
  std::uint32_t type;
  std::string name;
  std::uint64_t contentBytes;
};

MessageHeader receiveMessageHeader(int socket) {
  MessageHeader header{};
  header.type = receiveU32(socket);
  const std::uint32_t nameBytes = receiveU32(socket);
  header.contentBytes = receiveU64(socket);
  if (nameBytes > 4096)
    throw std::runtime_error("Server sent an invalid streaming file name");
  header.name.resize(nameBytes);
  if (nameBytes > 0)
    receiveAll(socket, header.name.data(), header.name.size());
  return header;
}

void sendAck(int socket, const std::string &name) {
  sendMessageHeader(socket, kAckMessage, name, 0);
}

void receiveAck(int socket, const std::string &expectedName) {
  const MessageHeader ack = receiveMessageHeader(socket);
  if (ack.type != kAckMessage || ack.contentBytes != 0 ||
      ack.name != expectedName)
    throw std::runtime_error("Client sent an invalid frame acknowledgement");
}

std::vector<std::filesystem::path>
listCompletedFrames(const std::filesystem::path &directory) {
  std::vector<std::filesystem::path> frames;
  std::error_code error;
  std::filesystem::directory_iterator entries(directory, error);
  if (error)
    throw std::runtime_error("Could not inspect frame spool '" +
                             directory.string() + "': " + error.message());

  for (const auto &entry : entries) {
    if (entry.is_regular_file(error) && !error &&
        entry.path().extension() == ".nbsnap")
      frames.push_back(entry.path());
    error.clear();
  }
  std::sort(frames.begin(), frames.end());
  return frames;
}

bool waitForFrameAcknowledgement(
    int socket, const std::string &name,
    const std::function<bool()> &stopRequested) {
  // Sending a large frame only guarantees that its bytes have entered the
  // socket pipeline. They may still be crossing the relay and SSH tunnel, and
  // the client must flush, validate, and publish the file before acknowledging
  // it. Do not start the socket's short receive timeout until ACK bytes are
  // actually readable.
  while (!stopRequested()) {
    if (!waitForReadable(socket, std::chrono::milliseconds(250)))
      continue;
    receiveAck(socket, name);
    return true;
  }
  return false;
}

bool sendFrame(int socket, const std::filesystem::path &framePath,
               const std::function<bool()> &stopRequested) {
  std::error_code error;
  const auto fileBytes = std::filesystem::file_size(framePath, error);
  if (error)
    throw std::runtime_error("Could not determine queued frame size: " +
                             error.message());

  std::ifstream input(framePath, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open queued frame: " +
                             framePath.string());

  const std::string name = framePath.filename().string();
  sendMessageHeader(socket, kFrameMessage, name, fileBytes);

  std::vector<char> buffer(kIoBufferBytes);
  std::uint64_t remaining = fileBytes;
  while (remaining > 0) {
    const auto wanted = static_cast<std::streamsize>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    input.read(buffer.data(), wanted);
    const auto read = input.gcount();
    if (read != wanted)
      throw std::runtime_error(
          "Queued frame changed while it was being sent: " +
          framePath.string());
    sendAll(socket, buffer.data(), static_cast<std::size_t>(read));
    remaining -= static_cast<std::uint64_t>(read);
  }
  if (!waitForFrameAcknowledgement(socket, name, stopRequested))
    return false;

  std::filesystem::remove(framePath, error);
  if (error)
    throw std::runtime_error("Client acknowledged frame, but it could not be "
                             "removed from the spool: " +
                             error.message());
  return true;
}

void receiveFrame(int socket, const MessageHeader &header,
                  const std::filesystem::path &outputDirectory) {
  const std::filesystem::path receivedName(header.name);
  if (header.name.empty() || receivedName.filename() != receivedName ||
      header.name == "." || header.name == ".." ||
      receivedName.extension() != ".nbsnap" || header.contentBytes == 0)
    throw std::runtime_error("Server offered an invalid frame envelope");

  const std::filesystem::path finalPath = outputDirectory / receivedName;
  std::filesystem::path temporaryPath = finalPath;
  temporaryPath += ".part";

  try {
    std::ofstream output(temporaryPath,
                         std::ios::binary | std::ios::trunc | std::ios::out);
    if (!output.is_open())
      throw std::runtime_error("Could not create received frame: " +
                               temporaryPath.string());

    std::vector<char> buffer(kIoBufferBytes);
    std::uint64_t remaining = header.contentBytes;
    while (remaining > 0) {
      const std::size_t bytes = static_cast<std::size_t>(
          std::min<std::uint64_t>(remaining, buffer.size()));
      receiveAll(socket, buffer.data(), bytes);
      output.write(buffer.data(), static_cast<std::streamsize>(bytes));
      if (!output)
        throw std::runtime_error("Failed while writing received frame: " +
                                 temporaryPath.string());
      remaining -= bytes;
    }

    output.flush();
    if (!output)
      throw std::runtime_error("Could not flush received frame: " +
                               temporaryPath.string());
    output.close();

    std::error_code error;
    std::filesystem::rename(temporaryPath, finalPath, error);
    if (error)
      throw std::runtime_error("Could not publish received frame: " +
                               error.message());
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporaryPath, ignored);
    throw;
  }
}

void publishLatestFrame(const std::filesystem::path &outputDirectory,
                        const std::string &frameName) {
  const std::filesystem::path marker = outputDirectory / ".nbody-latest";
  std::filesystem::path temporary = marker;
  temporary += ".part";

  try {
    std::ofstream output(temporary,
                         std::ios::out | std::ios::trunc | std::ios::binary);
    if (!output.is_open())
      throw std::runtime_error("Could not create latest-frame marker: " +
                               temporary.string());
    output << frameName << '\n';
    output.flush();
    if (!output)
      throw std::runtime_error("Could not write latest-frame marker: " +
                               temporary.string());
    output.close();

    std::error_code error;
    std::filesystem::rename(temporary, marker, error);
    if (error)
      throw std::runtime_error("Could not publish latest-frame marker: " +
                               error.message());
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

bool waitForReadable(int descriptor, std::chrono::milliseconds timeout) {
  pollfd descriptorState{descriptor, POLLIN, 0};
  const int result =
      ::poll(&descriptorState, 1, static_cast<int>(timeout.count()));
  if (result < 0 && errno == EINTR)
    return false;
  if (result < 0)
    throw std::runtime_error(socketError("Socket poll failed"));
  if (result == 0)
    return false;
  if ((descriptorState.revents & POLLIN) != 0)
    return true;
  if ((descriptorState.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
    throw std::runtime_error("Peer disconnected while waiting for data");
  return false;
}

void serveClient(int socket, const ServerConfig &config,
                 const std::function<bool()> &stopRequested,
                 std::ostream &log) {
  const bool follow = receiveHello(socket);
  log << "Client connected in " << (follow ? "follow" : "backlog") << " mode\n";

  // Backlog mode is a snapshot of the queue at connection time. Follow mode
  // rescans after every acknowledgement and naturally picks up new frames.
  std::vector<std::filesystem::path> frames =
      listCompletedFrames(config.spoolDirectory);
  std::size_t nextFrame = 0;

  while (!stopRequested()) {
    if (nextFrame < frames.size()) {
      const auto framePath = frames[nextFrame++];
      std::error_code error;
      if (!std::filesystem::exists(framePath, error) || error)
        continue;
      const std::string name = framePath.filename().string();
      if (!sendFrame(socket, framePath, stopRequested))
        return;
      log << "Acknowledged " << name << '\n';
      continue;
    }

    if (!follow) {
      sendMessageHeader(socket, kEndMessage, {}, 0);
      return;
    }

    frames = listCompletedFrames(config.spoolDirectory);
    nextFrame = 0;
    if (frames.empty() && waitForReadable(socket, config.pollInterval)) {
      char byte = 0;
      const ssize_t received = ::recv(socket, &byte, sizeof(byte), MSG_PEEK);
      if (received == 0)
        throw std::runtime_error("Peer disconnected");
      if (received < 0)
        throw std::runtime_error(socketError("Socket receive failed"));
      throw std::runtime_error("Client sent unexpected data while idle");
    }
  }
}

} // namespace

void runServer(const ServerConfig &config,
               const std::function<bool()> &stopRequested, std::ostream &log) {
  if (config.port == 0)
    throw std::invalid_argument("Streaming server port must be non-zero");
  if (config.pollInterval.count() <= 0)
    throw std::invalid_argument("Streaming poll interval must be positive");

  std::error_code error;
  std::filesystem::create_directories(config.spoolDirectory, error);
  if (error)
    throw std::runtime_error("Could not create frame spool directory: " +
                             error.message());

  Socket listener = createListener(config.bindAddress, config.port);
  log << "N-body stream server listening on " << config.bindAddress << ':'
      << config.port << "\nSpool: " << config.spoolDirectory << '\n';

  while (!stopRequested()) {
    if (!waitForReadable(listener.get(), std::chrono::milliseconds(250)))
      continue;

    sockaddr_storage peer{};
    socklen_t peerBytes = sizeof(peer);
    const int descriptor = ::accept(
        listener.get(), reinterpret_cast<sockaddr *>(&peer), &peerBytes);
    if (descriptor < 0 && errno == EINTR)
      continue;
    if (descriptor < 0)
      throw std::runtime_error(socketError("Could not accept client"));

    try {
      Socket client(descriptor);
      configureConnectedSocket(client.get());
      serveClient(client.get(), config, stopRequested, log);
    } catch (const std::exception &error) {
      if (!stopRequested())
        log << "Client disconnected: " << error.what() << '\n';
    }
  }
}

void runRelaySource(const ServerConfig &serverConfig,
                    const RelaySourceConfig &relayConfig,
                    const std::function<bool()> &stopRequested,
                    std::ostream &log) {
  if (serverConfig.pollInterval.count() <= 0)
    throw std::invalid_argument("Streaming poll interval must be positive");
  if (relayConfig.host.empty())
    throw std::invalid_argument("Relay host must not be empty");
  if (relayConfig.port == 0)
    throw std::invalid_argument("Relay port must be non-zero");
  if (relayConfig.reconnectInterval.count() <= 0)
    throw std::invalid_argument("Relay reconnect interval must be positive");

  std::error_code error;
  std::filesystem::create_directories(serverConfig.spoolDirectory, error);
  if (error)
    throw std::runtime_error("Could not create frame spool directory: " +
                             error.message());

  const std::string token = readRelayToken(relayConfig.tokenFile);
  log << "N-body stream source connecting to relay " << relayConfig.host << ':'
      << relayConfig.port << "\nSpool: " << serverConfig.spoolDirectory << '\n';

  while (!stopRequested()) {
    try {
      Socket relay = connectTo(relayConfig.host, relayConfig.port);
      sendRelayAuthentication(relay.get(), token);

      std::uint8_t authentication = kRelayRejected;
      receiveAll(relay.get(), &authentication, sizeof(authentication));
      if (authentication != kRelayAccepted)
        throw std::runtime_error("Relay rejected source authentication");

      log << "Authenticated with relay; waiting for a client\n";
      waitForRelayGo(relay.get(), stopRequested);
      if (stopRequested())
        return;

      log << "Relay paired a client\n";
      serveClient(relay.get(), serverConfig, stopRequested, log);
    } catch (const std::exception &error) {
      if (!stopRequested())
        log << "Relay connection interrupted: " << error.what()
            << "; reconnecting\n";
    }
    interruptibleDelay(relayConfig.reconnectInterval, stopRequested);
  }
}

void runRelay(const RelayConfig &config,
              const std::function<bool()> &stopRequested, std::ostream &log) {
  if (config.sourceBindAddress.empty() || config.clientBindAddress.empty())
    throw std::invalid_argument("Relay bind addresses must not be empty");
  if (config.sourcePort == 0 || config.clientPort == 0)
    throw std::invalid_argument("Relay ports must be non-zero");
  if (config.sourceBindAddress == config.clientBindAddress &&
      config.sourcePort == config.clientPort)
    throw std::invalid_argument(
        "Relay source and client listeners must use different endpoints");

  const std::string token = readRelayToken(config.tokenFile);
  Socket sourceListener =
      createListener(config.sourceBindAddress, config.sourcePort);
  Socket clientListener =
      createListener(config.clientBindAddress, config.clientPort);
  log << "N-body relay source listener on " << config.sourceBindAddress << ':'
      << config.sourcePort << "\nClient listener on "
      << config.clientBindAddress << ':' << config.clientPort << '\n';

  while (!stopRequested()) {
    if (!waitForReadable(sourceListener.get(), std::chrono::milliseconds(250)))
      continue;

    try {
      Socket source = acceptConnection(sourceListener.get());
      configureConnectedSocket(source.get());
      const bool authenticated =
          receiveRelayAuthentication(source.get(), token);
      const std::uint8_t status =
          authenticated ? kRelayAccepted : kRelayRejected;
      sendAll(source.get(), &status, sizeof(status));
      if (!authenticated) {
        log << "Rejected unauthenticated relay source\n";
        continue;
      }

      log << "Authenticated relay source; waiting for a client\n";
      Socket client =
          waitForRelayClient(clientListener.get(), source.get(), stopRequested);
      if (!client || stopRequested())
        return;

      configureConnectedSocket(client.get());
      sendAll(source.get(), &kRelayGo, sizeof(kRelayGo));
      log << "Paired relay source and client\n";
      bridgeConnections(source.get(), client.get(), stopRequested);
      if (!stopRequested())
        log << "Relay pair disconnected\n";
    } catch (const std::exception &error) {
      if (!stopRequested())
        log << "Relay session ended: " << error.what() << '\n';
    }
  }
}

std::size_t downloadFrames(const ClientConfig &config,
                           const std::function<bool()> &stopRequested,
                           std::ostream &log) {
  if (config.port == 0)
    throw std::invalid_argument("Streaming client port must be non-zero");

  std::error_code error;
  std::filesystem::create_directories(config.outputDirectory, error);
  if (error)
    throw std::runtime_error("Could not create client output directory: " +
                             error.message());

  Socket server = connectTo(config.host, config.port);
  sendHello(server.get(), config.follow);
  log << "Connected to " << config.host << ':' << config.port << '\n';

  std::size_t receivedFrames = 0;
  while (!stopRequested()) {
    if (!waitForReadable(server.get(), std::chrono::milliseconds(250)))
      continue;
    const MessageHeader header = receiveMessageHeader(server.get());
    if (header.type == kEndMessage) {
      if (!header.name.empty() || header.contentBytes != 0)
        throw std::runtime_error("Server sent an invalid end message");
      return receivedFrames;
    }
    if (header.type != kFrameMessage)
      throw std::runtime_error("Server sent an unknown message type");

    receiveFrame(server.get(), header, config.outputDirectory);
    (void)readFrameHeader(config.outputDirectory /
                          std::filesystem::path(header.name));
    publishLatestFrame(config.outputDirectory, header.name);
    sendAck(server.get(), header.name);
    ++receivedFrames;
    log << "Received " << header.name << " (" << header.contentBytes
        << " bytes)\n";
  }
  return receivedFrames;
}

} // namespace nbody::streaming
