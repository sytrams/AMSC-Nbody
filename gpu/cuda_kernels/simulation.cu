#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <cstddef>
#include <stdexcept>

#include <thrust/device_vector.h>

#include "simulation_cuda.hpp"

namespace nbody::detail {

struct SimulationDeviceState {
  explicit SimulationDeviceState(std::size_t count)
    : accelerationX(count, 0.0), accelerationY(count, 0.0),
      accelerationZ(count, 0.0), nextAccelerationX(count, 0.0),
      nextAccelerationY(count, 0.0), nextAccelerationZ(count, 0.0),
      mortonMass(count), mortonX(count), mortonY(count), mortonZ(count) {}

  thrust::device_vector<double> accelerationX;
  thrust::device_vector<double> accelerationY;
  thrust::device_vector<double> accelerationZ;
  thrust::device_vector<double> nextAccelerationX;
  thrust::device_vector<double> nextAccelerationY;
  thrust::device_vector<double> nextAccelerationZ;

  thrust::device_vector<double> mortonMass;
  thrust::device_vector<double> mortonX;
  thrust::device_vector<double> mortonY;
  thrust::device_vector<double> mortonZ;
};

__global__ void gatherMortonOrderedParticleDataKernel(
    const std::uint32_t *sortedIndices,
    const double *mass,
    const double *x,
    const double *y,
    const double *z,
    double *sortedMass,
    double *sortedX,
    double *sortedY,
    double *sortedZ,
    std::size_t particleCount) {

  const std::size_t sortedIndex =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (sortedIndex >= particleCount)
    return;

  const std::uint32_t originalIndex = sortedIndices[sortedIndex];

  sortedMass[sortedIndex] = mass[originalIndex];
  sortedX[sortedIndex] = x[originalIndex];
  sortedY[sortedIndex] = y[originalIndex];
  sortedZ[sortedIndex] = z[originalIndex];
}

SimulationDeviceState *createSimulationDeviceState(std::size_t particleCount) {
  if (particleCount == 0)
    throw std::invalid_argument("Simulation device state cannot be empty");

  return new SimulationDeviceState(particleCount);
}

void gatherMortonOrderedParticleData(
    SimulationDeviceState &state,
    const std::uint32_t *sortedIndices,
    const double *mass,
    const double *x,
    const double *y,
    const double *z,
    std::size_t particleCount) {

  if (sortedIndices == nullptr || mass == nullptr ||
      x == nullptr || y == nullptr || z == nullptr)
    throw std::invalid_argument(
        "Morton particle gather input arrays must not be null");

  if (particleCount != state.mortonMass.size())
    throw std::invalid_argument(
        "Morton particle gather count does not match simulation state");

  constexpr int threadsPerBlock = 256;

  const int blocks = static_cast<int>(
      (particleCount + threadsPerBlock - 1) / threadsPerBlock);

  gatherMortonOrderedParticleDataKernel<<<blocks, threadsPerBlock>>>(
      sortedIndices,
      mass,
      x,
      y,
      z,
      thrust::raw_pointer_cast(state.mortonMass.data()),
      thrust::raw_pointer_cast(state.mortonX.data()),
      thrust::raw_pointer_cast(state.mortonY.data()),
      thrust::raw_pointer_cast(state.mortonZ.data()),
      particleCount);

  const cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess)
    throw std::runtime_error(
        std::string("gatherMortonOrderedParticleDataKernel launch failed: ") +
        cudaGetErrorString(error));
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

DeviceMortonParticleView mortonOrderedParticles(SimulationDeviceState &state) noexcept {
  return {
      state.mortonMass.size(),
      thrust::raw_pointer_cast(state.mortonMass.data()),
      thrust::raw_pointer_cast(state.mortonX.data()),
      thrust::raw_pointer_cast(state.mortonY.data()),
      thrust::raw_pointer_cast(state.mortonZ.data())};
}

void swapAccelerationBuffers(SimulationDeviceState &state) noexcept {
  state.accelerationX.swap(state.nextAccelerationX);
  state.accelerationY.swap(state.nextAccelerationY);
  state.accelerationZ.swap(state.nextAccelerationZ);
}

} // namespace nbody::detail
