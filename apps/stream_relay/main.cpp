#include <charconv>
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

std::uint16_t parsePort(std::string_view text, std::string_view option) {
  unsigned int value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value == 0 || value > 65535)
    throw std::invalid_argument("Invalid port for " + std::string(option) +
                                ": " + std::string(text));
  return static_cast<std::uint16_t>(value);
}

void printUsage(std::ostream &output, const char *program) {
  output << "Usage: " << program << " --token-file FILE [OPTIONS]\n\n"
         << "  --token-file FILE       Private source authentication token\n"
         << "  --source-bind ADDRESS   Source listener (default: 0.0.0.0)\n"
         << "  --source-port PORT      Source listener port (default: 4748)\n"
         << "  --client-bind ADDRESS   Client listener (default: 127.0.0.1)\n"
         << "  --client-port PORT      Client listener port (default: 4747)\n"
         << "  -h, --help              Show this help and exit\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::cout << std::unitbuf;
    nbody::streaming::RelayConfig config;
    for (int index = 1; index < argc; ++index) {
      const std::string_view option(argv[index]);
      if (option == "-h" || option == "--help") {
        printUsage(std::cout, argv[0]);
        return 0;
      }
      if (option == "--token-file")
        config.tokenFile = requireValue(argc, argv, index, option);
      else if (option == "--source-bind")
        config.sourceBindAddress = requireValue(argc, argv, index, option);
      else if (option == "--source-port")
        config.sourcePort =
            parsePort(requireValue(argc, argv, index, option), option);
      else if (option == "--client-bind")
        config.clientBindAddress = requireValue(argc, argv, index, option);
      else if (option == "--client-port")
        config.clientPort =
            parsePort(requireValue(argc, argv, index, option), option);
      else
        throw std::invalid_argument("Unknown option: " + std::string(option));
    }

    if (config.tokenFile.empty())
      throw std::invalid_argument("--token-file is required");

    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    nbody::streaming::runRelay(
        config, [] { return stopRequested != 0; }, std::cout);
    return 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Streaming relay failed: " << error.what() << '\n';
    return 1;
  }
}
