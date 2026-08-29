#pragma once

#include <filesystem>
#include <iosfwd>
#include <optional>

#include "streaming/network.hpp"

namespace nbody::streaming {

// Owns the streaming server started for a --cluster simulation. The child is
// stopped when the simulation exits; queued files remain on disk and can be
// served later by launching nbody_stream_server manually.
class ServerProcess {
public:
  ServerProcess(const std::filesystem::path &executable,
                const ServerConfig &config,
                const std::optional<RelaySourceConfig> &relayConfig,
                std::ostream &log);
  ~ServerProcess() noexcept;

  ServerProcess(const ServerProcess &) = delete;
  ServerProcess &operator=(const ServerProcess &) = delete;

private:
  int processId_ = -1;
};

[[nodiscard]] std::filesystem::path
defaultServerExecutable(const std::filesystem::path &simulationExecutable);

} // namespace nbody::streaming
