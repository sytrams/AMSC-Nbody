#include "streaming/device_capture.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace nbody::streaming {
namespace {

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

__global__ void gatherPositionSamples(const double *x, const double *y,
                                      const double *z,
                                      std::size_t sourceParticleCount,
                                      std::size_t sampleParticleCount,
                                      std::size_t sampleOffset,
                                      std::size_t chunkParticleCount,
                                      float *positions) {
  const std::size_t chunkIndex =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunkIndex >= chunkParticleCount)
    return;

  const std::size_t sampleIndex = sampleOffset + chunkIndex;
  const std::size_t sourceIndex =
      (sampleIndex * sourceParticleCount) / sampleParticleCount;
  positions[chunkIndex * 3] = static_cast<float>(x[sourceIndex]);
  positions[chunkIndex * 3 + 1] = static_cast<float>(y[sourceIndex]);
  positions[chunkIndex * 3 + 2] = static_cast<float>(z[sourceIndex]);
}

__global__ void gatherTypeSamples(const std::uint8_t *particleTypes,
                                  std::size_t sourceParticleCount,
                                  std::size_t sampleParticleCount,
                                  std::size_t sampleOffset,
                                  std::size_t chunkParticleCount,
                                  std::uint8_t *types) {
  const std::size_t chunkIndex =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunkIndex >= chunkParticleCount)
    return;

  const std::size_t sampleIndex = sampleOffset + chunkIndex;
  const std::size_t sourceIndex =
      (sampleIndex * sourceParticleCount) / sampleParticleCount;
  types[chunkIndex] = particleTypes[sourceIndex];
}

} // namespace

DeviceFrameWriter::DeviceFrameWriter(ConstDeviceParticlesView particles,
                                     std::size_t maximumParticles)
    : particles_(particles),
      sampleParticleCount_(maximumParticles == 0
                               ? particles.count
                               : std::min(particles.count, maximumParticles)) {
  if (particles_.count == 0 || particles_.x == nullptr ||
      particles_.y == nullptr || particles_.z == nullptr ||
      particles_.type == nullptr) {
    throw std::invalid_argument(
        "Cannot capture an empty or incomplete device particle view");
  }

  checkCuda(cudaMalloc(reinterpret_cast<void **>(&devicePositions_),
                       std::min(kFrameChunkParticles, sampleParticleCount_) *
                           kPositionComponents * sizeof(float)),
            "Could not allocate the graphical sampling buffer");
  const cudaError_t typeAllocation = cudaMalloc(
      reinterpret_cast<void **>(&deviceTypes_),
      std::min(kFrameChunkParticles, sampleParticleCount_) *
          sizeof(std::uint8_t));
  if (typeAllocation != cudaSuccess) {
    (void)cudaFree(devicePositions_);
    devicePositions_ = nullptr;
    checkCuda(typeAllocation,
              "Could not allocate the graphical particle-type buffer");
  }
}

DeviceFrameWriter::~DeviceFrameWriter() noexcept {
  if (deviceTypes_ != nullptr)
    (void)cudaFree(deviceTypes_);
  if (devicePositions_ != nullptr)
    (void)cudaFree(devicePositions_);
}

std::size_t DeviceFrameWriter::sourceParticleCount() const noexcept {
  return particles_.count;
}

std::size_t DeviceFrameWriter::sampleParticleCount() const noexcept {
  return sampleParticleCount_;
}

std::filesystem::path
DeviceFrameWriter::write(const FrameSpool &spool, std::uint64_t sequence,
                         std::uint64_t simulationStep,
                         double simulationTime,
                         std::uint64_t totalSteps) const {

  const PositionChunkReader reader =
      [this](std::size_t offset, std::size_t count, float *xyz) {
        constexpr unsigned int threads = 256;
        const auto blocks = static_cast<unsigned int>((count + threads - 1) /
                                                       threads);
        gatherPositionSamples<<<blocks, threads>>>(
            particles_.x, particles_.y, particles_.z, particles_.count,
            sampleParticleCount_, offset, count, devicePositions_);
        checkCuda(cudaGetLastError(),
                  "Could not launch the graphical sampling kernel");
        checkCuda(cudaMemcpy(xyz, devicePositions_,
                             count * kPositionComponents * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "Could not copy graphical samples to the frame writer");
      };

  const ParticleTypeChunkReader typeReader =
      [this](std::size_t offset, std::size_t count, std::uint8_t *types) {
        constexpr unsigned int threads = 256;
        const auto blocks = static_cast<unsigned int>((count + threads - 1) /
                                                       threads);
        gatherTypeSamples<<<blocks, threads>>>(
            particles_.type, particles_.count, sampleParticleCount_, offset,
            count, deviceTypes_);
        checkCuda(cudaGetLastError(),
                  "Could not launch the graphical particle-type kernel");
        checkCuda(cudaMemcpy(types, deviceTypes_, count * sizeof(std::uint8_t),
                             cudaMemcpyDeviceToHost),
                  "Could not copy graphical particle types to the frame writer");
      };

  return spool.writeTypedPositions(sequence, simulationStep, simulationTime,
                                   totalSteps, sampleParticleCount_, reader,
                                   typeReader, particles_.count);
}

} // namespace nbody::streaming
