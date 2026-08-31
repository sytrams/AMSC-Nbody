#pragma once

#include <cstddef>
#include <cstdint>

namespace nbody::detail {

struct DeviceAccelerationView {
  std::size_t count = 0;
  double *x = nullptr;
  double *y = nullptr;
  double *z = nullptr;
};

struct DeviceMortonParticleView {
  std::size_t count = 0;
  const double *mass = nullptr;
  const double *x = nullptr;
  const double *y = nullptr;
  const double *z = nullptr;
};

struct SimulationDeviceState;

[[nodiscard]] SimulationDeviceState *
createSimulationDeviceState(std::size_t particleCount);
void destroySimulationDeviceState(SimulationDeviceState *state) noexcept;

[[nodiscard]] DeviceAccelerationView
currentAcceleration(SimulationDeviceState &state) noexcept;
[[nodiscard]] DeviceAccelerationView
nextAcceleration(SimulationDeviceState &state) noexcept;

void swapAccelerationBuffers(SimulationDeviceState &state) noexcept;

void gatherMortonOrderedParticleData(
    SimulationDeviceState &state,
    const std::uint32_t *sortedIndices,
    const double *mass,
    const double *x,
    const double *y,
    const double *z,
    std::size_t particleCount);

[[nodiscard]] DeviceMortonParticleView
mortonOrderedParticles(SimulationDeviceState &state) noexcept;

} // namespace nbody::detail
