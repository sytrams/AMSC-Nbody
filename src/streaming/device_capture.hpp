#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

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
  std::uint16_t *devicePositions_ = nullptr;
  double *deviceBlockBounds_ = nullptr;
  int *deviceInvalidPosition_ = nullptr;
  std::vector<std::uint8_t> sampledTypes_;
};

} // namespace nbody::streaming
