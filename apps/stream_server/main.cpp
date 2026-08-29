#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "streaming/network.hpp"

namespace {

volatile std::sig_atomic_t stopRequested = 0;

void requestStop(int) { stopRequested = 1; }

std::string_view requireValue(int argc, char **argv, int &index,
                              std::string_view option) {
  if (index + 1 >= argc)
    throw std::invalid_argument("Missing value for " + std::string(option));
  return argv[++index];
}

std::uint16_t parsePort(std::string_view text) {
  unsigned int value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value == 0 || value > 65535)
    throw std::invalid_argument("Invalid TCP port: " + std::string(text));
  return static_cast<std::uint16_t>(value);
}

long parseMilliseconds(std::string_view text) {
  long value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value <= 0 || value > 60000)
    throw std::invalid_argument("Invalid poll interval: " + std::string(text));
  return value;
}

void printUsage(std::ostream &output, const char *program) {
  output << "Usage: " << program << " [OPTIONS]\n\n"
         << "  --spool-dir DIR  Durable frame queue (default: nbody-stream)\n"
         << "  --bind ADDRESS   Listen address (default: 127.0.0.1)\n"
         << "  --port PORT      TCP port (default: 4747)\n"
         << "  --poll-ms MS     New-frame poll interval (default: 100)\n"
         << "\nOutbound relay mode:\n"
         << "  --relay-host HOST       Connect outward instead of listening\n"
         << "  --relay-port PORT       Relay source port (default: 4748)\n"
         << "  --relay-token-file FILE Private shared authentication token\n"
         << "  --reconnect-ms MS       Relay reconnect delay (default: 1000)\n"
         << "  -h, --help       Show this help and exit\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::cout << std::unitbuf;
    nbody::streaming::ServerConfig config;
    nbody::streaming::RelaySourceConfig relayConfig;
    bool relayOptionSeen = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (option == "-h" || option == "--help") {
        printUsage(std::cout, argv[0]);
        return 0;
      }
      if (option == "--spool-dir")
        config.spoolDirectory = requireValue(argc, argv, index, option);
      else if (option == "--bind")
        config.bindAddress = requireValue(argc, argv, index, option);
      else if (option == "--port")
        config.port = parsePort(requireValue(argc, argv, index, option));
      else if (option == "--poll-ms")
        config.pollInterval = std::chrono::milliseconds(
            parseMilliseconds(requireValue(argc, argv, index, option)));
      else if (option == "--relay-host") {
        relayConfig.host = requireValue(argc, argv, index, option);
        relayOptionSeen = true;
      } else if (option == "--relay-port") {
        relayConfig.port = parsePort(requireValue(argc, argv, index, option));
        relayOptionSeen = true;
      } else if (option == "--relay-token-file") {
        relayConfig.tokenFile = requireValue(argc, argv, index, option);
        relayOptionSeen = true;
      } else if (option == "--reconnect-ms") {
        relayConfig.reconnectInterval = std::chrono::milliseconds(
            parseMilliseconds(requireValue(argc, argv, index, option)));
        relayOptionSeen = true;
      } else
        throw std::invalid_argument("Unknown option: " + std::string(option));
    }

    if (relayOptionSeen && relayConfig.host.empty())
      throw std::invalid_argument(
          "--relay-host is required when relay options are used");
    if (!relayConfig.host.empty() && relayConfig.tokenFile.empty())
      throw std::invalid_argument(
          "--relay-token-file is required in outbound relay mode");

    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    if (relayConfig.host.empty()) {
      nbody::streaming::runServer(
          config, [] { return stopRequested != 0; }, std::cout);
    } else {
      nbody::streaming::runRelaySource(
          config, relayConfig, [] { return stopRequested != 0; }, std::cout);
    }
    return 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Streaming server failed: " << error.what() << '\n';
    return 1;
  }
}
