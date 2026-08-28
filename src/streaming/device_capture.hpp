#pragma once

#include <cstdint>
#include <filesystem>

#include "particle.hpp"
#include "streaming/frame.hpp"

namespace nbody::streaming {

class DeviceFrameWriter {
public:
  DeviceFrameWriter(ConstDeviceParticlesView particles,
                    std::size_t maximumParticles);
  ~DeviceFrameWriter() noexcept;

  DeviceFrameWriter(const DeviceFrameWriter &) = delete;
  DeviceFrameWriter &operator=(const DeviceFrameWriter &) = delete;

  [[nodiscard]] std::size_t sourceParticleCount() const noexcept;
  [[nodiscard]] std::size_t sampleParticleCount() const noexcept;

  std::filesystem::path write(const FrameSpool &spool,
                              std::uint64_t sequence,
                              std::uint64_t simulationStep,
                              double simulationTime) const;

private:
  ConstDeviceParticlesView particles_{};
  std::size_t sampleParticleCount_ = 0;
  float *devicePositions_ = nullptr;
};

} // namespace nbody::streaming
