#include <cstddef>
#include <stdexcept>

#include <thrust/device_vector.h>

#include "simulation_cuda.hpp"

namespace nbody::detail {

struct SimulationDeviceState {
  explicit SimulationDeviceState(std::size_t count)
      : accelerationX(count, 0.0), accelerationY(count, 0.0),
        accelerationZ(count, 0.0), nextAccelerationX(count, 0.0),
        nextAccelerationY(count, 0.0), nextAccelerationZ(count, 0.0) {}

  thrust::device_vector<double> accelerationX;
  thrust::device_vector<double> accelerationY;
  thrust::device_vector<double> accelerationZ;
  thrust::device_vector<double> nextAccelerationX;
  thrust::device_vector<double> nextAccelerationY;
  thrust::device_vector<double> nextAccelerationZ;
};

SimulationDeviceState *createSimulationDeviceState(std::size_t particleCount) {
  if (particleCount == 0)
    throw std::invalid_argument("Simulation device state cannot be empty");

  return new SimulationDeviceState(particleCount);
}

void destroySimulationDeviceState(SimulationDeviceState *state) noexcept {
  delete state;
}

DeviceAccelerationView
currentAcceleration(SimulationDeviceState &state) noexcept {
  return {state.accelerationX.size(),
          thrust::raw_pointer_cast(state.accelerationX.data()),
          thrust::raw_pointer_cast(state.accelerationY.data()),
          thrust::raw_pointer_cast(state.accelerationZ.data())};
}

DeviceAccelerationView nextAcceleration(SimulationDeviceState &state) noexcept {
  return {state.nextAccelerationX.size(),
          thrust::raw_pointer_cast(state.nextAccelerationX.data()),
          thrust::raw_pointer_cast(state.nextAccelerationY.data()),
          thrust::raw_pointer_cast(state.nextAccelerationZ.data())};
}

void swapAccelerationBuffers(SimulationDeviceState &state) noexcept {
  state.accelerationX.swap(state.nextAccelerationX);
  state.accelerationY.swap(state.nextAccelerationY);
  state.accelerationZ.swap(state.nextAccelerationZ);
}

} // namespace nbody::detail
