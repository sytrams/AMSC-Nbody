#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

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

unsigned long parsePositive(std::string_view text, std::string_view option,
                            unsigned long maximum) {
  unsigned long value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value == 0 || value > maximum)
    throw std::invalid_argument("Invalid value for " + std::string(option) +
                                ": " + std::string(text));
  return value;
}

void printUsage(std::ostream &output, const char *program) {
  output << "Usage: " << program << " [OPTIONS]\n\n"
         << "  --host ADDRESS    Server address (default: 127.0.0.1)\n"
         << "  --port PORT       TCP port (default: 4747)\n"
         << "  --output-dir DIR  Destination for complete frames "
            "(default: nbody-frames)\n"
         << "  --once            Download the current backlog and exit\n"
         << "  --reconnect-ms MS Follow-mode reconnect delay (default: 1000)\n"
         << "  -h, --help        Show this help and exit\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    nbody::streaming::ClientConfig config;
    std::chrono::milliseconds reconnectDelay{1000};

    for (int index = 1; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (option == "-h" || option == "--help") {
        printUsage(std::cout, argv[0]);
        return 0;
      }
      if (option == "--host")
        config.host = requireValue(argc, argv, index, option);
      else if (option == "--port")
        config.port = static_cast<std::uint16_t>(parsePositive(
            requireValue(argc, argv, index, option), option, 65535));
      else if (option == "--output-dir")
        config.outputDirectory = requireValue(argc, argv, index, option);
      else if (option == "--once")
        config.follow = false;
      else if (option == "--reconnect-ms")
        reconnectDelay = std::chrono::milliseconds(parsePositive(
            requireValue(argc, argv, index, option), option, 60000));
      else
        throw std::invalid_argument("Unknown option: " + std::string(option));
    }

    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif

    if (!config.follow) {
      const std::size_t received = nbody::streaming::downloadFrames(
          config, [] { return stopRequested != 0; }, std::cout);
      std::cout << "Received " << received << " frame(s)\n";
      return 0;
    }

    while (!stopRequested) {
      try {
        (void)nbody::streaming::downloadFrames(
            config, [] { return stopRequested != 0; }, std::cout);
        if (stopRequested)
          break;
      } catch (const std::exception &error) {
        if (stopRequested)
          break;
        std::cerr << "Stream interrupted: " << error.what()
                  << "; reconnecting\n";
      }
      std::this_thread::sleep_for(reconnectDelay);
    }

    std::cout << "Streaming client stopped\n";
    return 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Streaming client failed: " << error.what() << '\n';
    return 1;
  }
}
