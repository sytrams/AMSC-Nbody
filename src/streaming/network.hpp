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

struct RelaySourceConfig {
  std::string host;
  std::uint16_t port = 4748;
  std::filesystem::path tokenFile;
  std::chrono::milliseconds reconnectInterval{1000};
};

struct RelayConfig {
  std::string sourceBindAddress = "0.0.0.0";
  std::uint16_t sourcePort = 4748;
  std::string clientBindAddress = "127.0.0.1";
  std::uint16_t clientPort = 4747;
  std::filesystem::path tokenFile;
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

// Connects outward to an authenticated relay, then serves the durable spool
// over each paired client connection. Failed and unavailable relay connections
// are retried without removing queued frames.
void runRelaySource(const ServerConfig &serverConfig,
                    const RelaySourceConfig &relayConfig,
                    const std::function<bool()> &stopRequested,
                    std::ostream &log);

// Pairs one authenticated outbound source with one client and forwards the
// existing frame protocol byte-for-byte. The client listener defaults to
// loopback so it can be exposed safely through SSH.
void runRelay(const RelayConfig &config,
              const std::function<bool()> &stopRequested, std::ostream &log);

// Runs one connection. A follow=false session returns after the backlog that
// existed when the client connected has been acknowledged. A follow=true
// session waits for new frames until stopped or disconnected.
std::size_t downloadFrames(const ClientConfig &config,
                           const std::function<bool()> &stopRequested,
                           std::ostream &log);

} // namespace nbody::streaming
