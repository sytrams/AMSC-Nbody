#pragma once

#include <cstddef>

namespace nbody::detail {

struct DeviceAccelerationView {
  std::size_t count = 0;
  double *x = nullptr;
  double *y = nullptr;
  double *z = nullptr;
};

struct SimulationDeviceState;

[[nodiscard]] SimulationDeviceState *
createSimulationDeviceState(std::size_t particleCount);
void destroySimulationDeviceState(SimulationDeviceState *state) noexcept;

[[nodiscard]] DeviceAccelerationView
currentAcceleration(SimulationDeviceState &state) noexcept;
[[nodiscard]] DeviceAccelerationView
nextAcceleration(SimulationDeviceState &state) noexcept;

void clearNextAcceleration(SimulationDeviceState &state);
void swapAccelerationBuffers(SimulationDeviceState &state) noexcept;

} // namespace nbody::detail
