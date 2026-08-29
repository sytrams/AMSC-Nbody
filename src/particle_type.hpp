#pragma once

#include <cstdint>

// These values are part of the on-disk particle format. Keep them synchronized
// with data/particle_types.py and apps/viewer/renderer.metal.
enum class ParticleType : std::uint8_t {
  Unknown = 0,
  Star = 1,
  Planet = 2,
  Moon = 3,
  Asteroid = 4,
};

[[nodiscard]] inline constexpr bool
isValidParticleType(std::uint8_t value) noexcept {
  return value <= static_cast<std::uint8_t>(ParticleType::Asteroid);
}
