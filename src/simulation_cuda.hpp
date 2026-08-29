#pragma once

#include <cstddef>
#include <memory>

namespace nbody::detail {

struct DeviceAccelerationView {
  std::size_t count = 0;
  double *x = nullptr;
  double *y = nullptr;
  double *z = nullptr;
};

struct SimulationDeviceState;

struct SimulationDeviceStateDeleter {
  void operator()(SimulationDeviceState *state) const noexcept;
};

using SimulationDeviceStatePtr =
    std::unique_ptr<SimulationDeviceState, SimulationDeviceStateDeleter>;

[[nodiscard]] SimulationDeviceStatePtr
createSimulationDeviceState(std::size_t particleCount);

[[nodiscard]] DeviceAccelerationView
currentAcceleration(SimulationDeviceState &state) noexcept;
[[nodiscard]] DeviceAccelerationView
nextAcceleration(SimulationDeviceState &state) noexcept;

void clearNextAcceleration(SimulationDeviceState &state);
void swapAccelerationBuffers(SimulationDeviceState &state) noexcept;

} // namespace nbody::detail
