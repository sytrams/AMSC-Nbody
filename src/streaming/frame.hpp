#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace nbody::streaming {

inline constexpr std::uint32_t kLegacyFrameVersion = 1;
inline constexpr std::uint32_t kQuantizedFrameVersion = 2;
inline constexpr std::uint32_t kFrameVersion = 3;
inline constexpr std::size_t kLegacyFrameHeaderBytes = 72;
inline constexpr std::size_t kQuantizedFrameHeaderBytes = 120;
inline constexpr std::size_t kFrameHeaderBytes = 128;
inline constexpr std::size_t kPositionComponents = 3;
inline constexpr std::size_t kFrameChunkParticles = 1024 * 1024;
inline constexpr std::uint16_t kQuantizedPositionMaximum = 65535;

struct PositionBounds {
  std::array<double, kPositionComponents> minimum{};
  std::array<double, kPositionComponents> maximum{};
};

struct FrameHeader {
  std::uint32_t version = kFrameVersion;
  std::uint32_t headerBytes = static_cast<std::uint32_t>(kFrameHeaderBytes);
  std::uint64_t sequence = 0;
  std::uint64_t simulationStep = 0;
  double simulationTime = 0.0;
  std::uint64_t sourceParticleCount = 0;
  std::uint64_t particleCount = 0;
  std::uint32_t scalarBytes = sizeof(std::uint16_t);
  std::uint32_t components = kPositionComponents;
  std::uint64_t payloadBytes = 0;
  PositionBounds bounds{};
  std::uint64_t particleTypeBytes = 0;
};

// The callback fills `count * 3` interleaved, independently quantized xyz
// values beginning at the requested particle offset. FrameSpool invokes it in
// bounded chunks so large simulations need no second full-size host copy.
using QuantizedPositionChunkReader = std::function<void(
    std::size_t offset, std::size_t count, std::uint16_t *xyz)>;

// Particle types use the stable values from particle_type.hpp. They are stored
// after the position payload so every frame is independently renderable.
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

  std::filesystem::path writeQuantizedPositions(
      std::uint64_t sequence, std::uint64_t simulationStep,
      double simulationTime, std::size_t particleCount,
      const PositionBounds &bounds,
      const QuantizedPositionChunkReader &reader,
      std::size_t sourceParticleCount = 0,
      const ParticleTypeChunkReader &typeReader = {}) const;

  // Atomically publishes a marker after the final frame has been written. A
  // viewer uses this to switch from following the newest frame to loop mode.
  [[nodiscard]] std::filesystem::path markComplete() const;

private:
  std::filesystem::path directory_;
  std::string sessionId_;
};

[[nodiscard]] FrameHeader
readFrameHeader(const std::filesystem::path &framePath);

// Returns interleaved float32 xyz values. Version-2/3 integer payloads are
// dequantized from their per-axis bounds; legacy version-1 floats are copied.
[[nodiscard]] std::vector<float>
readFramePositions(const std::filesystem::path &framePath);

// Returns one stable particle type per sampled particle. Older position-only
// frames are represented as Unknown so consumers remain backward compatible.
[[nodiscard]] std::vector<std::uint8_t>
readFrameParticleTypes(const std::filesystem::path &framePath);

} // namespace nbody::streaming
