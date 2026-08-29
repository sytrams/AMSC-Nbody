#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>

namespace nbody::streaming {

struct ServerConfig {
  std::filesystem::path spoolDirectory = "nbody-stream";
  std::string bindAddress = "127.0.0.1";
  std::uint16_t port = 4747;
  std::chrono::milliseconds pollInterval{100};
};

struct ClientConfig {
  std::string host = "127.0.0.1";
  std::uint16_t port = 4747;
  std::filesystem::path outputDirectory = "nbody-frames";
  bool follow = true;
};

// Serves one client at a time. Completed spool files are removed only after a
// client has validated the whole file and acknowledged its exact name.
void runServer(const ServerConfig &config,
               const std::function<bool()> &stopRequested, std::ostream &log);

// Runs one connection. A follow=false session returns after the backlog that
// existed when the client connected has been acknowledged. A follow=true
// session waits for new frames until stopped or disconnected.
std::size_t downloadFrames(const ClientConfig &config,
                           const std::function<bool()> &stopRequested,
                           std::ostream &log);

} // namespace nbody::streaming
