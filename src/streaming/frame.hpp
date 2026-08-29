#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace nbody::streaming {

inline constexpr std::size_t kLegacyFrameHeaderBytes = 72;
inline constexpr std::size_t kFrameHeaderBytes = 80;
inline constexpr std::size_t kPositionComponents = 3;
inline constexpr std::size_t kFrameChunkParticles = 1024 * 1024;

struct FrameHeader {
  std::uint32_t version = 2;
  std::uint32_t headerBytes = kFrameHeaderBytes;
  std::uint64_t sequence = 0;
  std::uint64_t simulationStep = 0;
  double simulationTime = 0.0;
  std::uint64_t sourceParticleCount = 0;
  std::uint64_t particleCount = 0;
  std::uint32_t scalarBytes = sizeof(float);
  std::uint32_t components = kPositionComponents;
  std::uint64_t payloadBytes = 0;
  std::uint64_t totalSteps = 0;
};

// The callback fills `count * 3` interleaved xyz floats beginning at the
// requested particle offset. FrameSpool invokes it in bounded chunks so large
// simulations do not need a second full-size host copy of every position.
using PositionChunkReader =
    std::function<void(std::size_t offset, std::size_t count, float *xyz)>;

using ParticleTypeChunkReader = std::function<void(
    std::size_t offset, std::size_t count, std::uint8_t *types)>;

class FrameSampler {
public:
  explicit FrameSampler(double samplesPerSecond = 60.0);

  [[nodiscard]] bool isDue(double elapsedWallSeconds);
  [[nodiscard]] std::uint64_t takeSequence() noexcept;
  [[nodiscard]] double rate() const noexcept;

private:
  double rate_;
  double interval_;
  double nextSampleTime_ = 0.0;
  std::uint64_t sequence_ = 0;
};

class FrameSpool {
public:
  explicit FrameSpool(std::filesystem::path directory,
                      std::string sessionId = {});

  [[nodiscard]] const std::filesystem::path &directory() const noexcept;
  [[nodiscard]] const std::string &sessionId() const noexcept;

  std::filesystem::path writePositions(std::uint64_t sequence,
                                       std::uint64_t simulationStep,
                                       double simulationTime,
                                       std::size_t particleCount,
                                       const PositionChunkReader &reader,
                                       std::size_t sourceParticleCount = 0) const;

  std::filesystem::path writeTypedPositions(
      std::uint64_t sequence, std::uint64_t simulationStep,
      double simulationTime, std::uint64_t totalSteps,
      std::size_t particleCount, const PositionChunkReader &positionReader,
      const ParticleTypeChunkReader &typeReader,
      std::size_t sourceParticleCount = 0) const;

private:
  std::filesystem::path directory_;
  std::string sessionId_;
};

[[nodiscard]] FrameHeader
readFrameHeader(const std::filesystem::path &framePath);

} // namespace nbody::streaming
