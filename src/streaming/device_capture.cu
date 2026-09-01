#include "streaming/device_capture.hpp"

#include "particle_type.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

namespace nbody::streaming {
namespace {

constexpr unsigned int kCaptureThreads = 256;
constexpr std::size_t kBoundsValuesPerBlock = kPositionComponents * 2;

void checkCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

__global__ void gatherPositionBounds(
    const double *x, const double *y, const double *z,
    std::size_t sourceParticleCount, std::size_t sampleParticleCount,
    std::size_t sampleOffset, std::size_t chunkParticleCount,
    double *blockBounds, int *invalidPosition) {
  extern __shared__ double sharedBounds[];
  double *minimumX = sharedBounds;
  double *minimumY = minimumX + blockDim.x;
  double *minimumZ = minimumY + blockDim.x;
  double *maximumX = minimumZ + blockDim.x;
  double *maximumY = maximumX + blockDim.x;
  double *maximumZ = maximumY + blockDim.x;

  const std::size_t chunkIndex =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  double positionX = CUDART_INF;
  double positionY = CUDART_INF;
  double positionZ = CUDART_INF;
  double upperX = -CUDART_INF;
  double upperY = -CUDART_INF;
  double upperZ = -CUDART_INF;

  if (chunkIndex < chunkParticleCount) {
    const std::size_t sampleIndex = sampleOffset + chunkIndex;
    const std::size_t sourceIndex =
        (sampleIndex * sourceParticleCount) / sampleParticleCount;
    positionX = x[sourceIndex];
    positionY = y[sourceIndex];
    positionZ = z[sourceIndex];
    if (::isfinite(positionX) && ::isfinite(positionY) &&
        ::isfinite(positionZ)) {
      upperX = positionX;
      upperY = positionY;
      upperZ = positionZ;
    } else {
      atomicExch(invalidPosition, 1);
      positionX = CUDART_INF;
      positionY = CUDART_INF;
      positionZ = CUDART_INF;
    }
  }

  minimumX[threadIdx.x] = positionX;
  minimumY[threadIdx.x] = positionY;
  minimumZ[threadIdx.x] = positionZ;
  maximumX[threadIdx.x] = upperX;
  maximumY[threadIdx.x] = upperY;
  maximumZ[threadIdx.x] = upperZ;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      minimumX[threadIdx.x] =
          ::fmin(minimumX[threadIdx.x], minimumX[threadIdx.x + stride]);
      minimumY[threadIdx.x] =
          ::fmin(minimumY[threadIdx.x], minimumY[threadIdx.x + stride]);
      minimumZ[threadIdx.x] =
          ::fmin(minimumZ[threadIdx.x], minimumZ[threadIdx.x + stride]);
      maximumX[threadIdx.x] =
          ::fmax(maximumX[threadIdx.x], maximumX[threadIdx.x + stride]);
      maximumY[threadIdx.x] =
          ::fmax(maximumY[threadIdx.x], maximumY[threadIdx.x + stride]);
      maximumZ[threadIdx.x] =
          ::fmax(maximumZ[threadIdx.x], maximumZ[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const std::size_t output =
        static_cast<std::size_t>(blockIdx.x) * kBoundsValuesPerBlock;
    blockBounds[output] = minimumX[0];
    blockBounds[output + 1] = minimumY[0];
    blockBounds[output + 2] = minimumZ[0];
    blockBounds[output + 3] = maximumX[0];
    blockBounds[output + 4] = maximumY[0];
    blockBounds[output + 5] = maximumZ[0];
  }
}

__device__ std::uint16_t quantizePosition(double position, double minimum,
                                          double maximum) {
  if (maximum == minimum)
    return 0;
  double normalized = (position - minimum) / (maximum - minimum);
  normalized = ::fmin(1.0, ::fmax(0.0, normalized));
  return static_cast<std::uint16_t>(
      normalized * static_cast<double>(kQuantizedPositionMaximum) + 0.5);
}

__global__ void gatherQuantizedPositionSamples(
    const double *x, const double *y, const double *z,
    std::size_t sourceParticleCount, std::size_t sampleParticleCount,
    std::size_t sampleOffset, std::size_t chunkParticleCount, double minimumX,
    double minimumY, double minimumZ, double maximumX, double maximumY,
    double maximumZ, std::uint16_t *positions) {
  const std::size_t chunkIndex =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunkIndex >= chunkParticleCount)
    return;

  const std::size_t sampleIndex = sampleOffset + chunkIndex;
  const std::size_t sourceIndex =
      (sampleIndex * sourceParticleCount) / sampleParticleCount;
  positions[chunkIndex * 3] =
      quantizePosition(x[sourceIndex], minimumX, maximumX);
  positions[chunkIndex * 3 + 1] =
      quantizePosition(y[sourceIndex], minimumY, maximumY);
  positions[chunkIndex * 3 + 2] =
      quantizePosition(z[sourceIndex], minimumZ, maximumZ);
}

__global__ void gatherParticleTypeSamples(
    const std::uint8_t *types, std::size_t sourceParticleCount,
    std::size_t sampleParticleCount, std::size_t sampleOffset,
    std::size_t chunkParticleCount, std::uint8_t *sampledTypes) {
  const std::size_t chunkIndex =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (chunkIndex >= chunkParticleCount)
    return;
  const std::size_t sampleIndex = sampleOffset + chunkIndex;
  const std::size_t sourceIndex =
      (sampleIndex * sourceParticleCount) / sampleParticleCount;
  sampledTypes[chunkIndex] = types[sourceIndex];
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

  const std::size_t chunkParticles =
      std::min(kFrameChunkParticles, sampleParticleCount_);
  const std::size_t boundsBlocks =
      (chunkParticles + kCaptureThreads - 1) / kCaptureThreads;
  std::uint8_t *deviceSampledTypes = nullptr;
  try {
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&devicePositions_),
                         chunkParticles * kPositionComponents *
                             sizeof(std::uint16_t)),
              "Could not allocate the quantized sampling buffer");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceBlockBounds_),
                         boundsBlocks * kBoundsValuesPerBlock * sizeof(double)),
              "Could not allocate the sampling bounds buffer");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceInvalidPosition_),
                         sizeof(int)),
              "Could not allocate the sampling validation flag");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceSampledTypes),
                         chunkParticles * sizeof(std::uint8_t)),
              "Could not allocate the particle type sampling buffer");

    sampledTypes_.resize(sampleParticleCount_);
    for (std::size_t offset = 0; offset < sampleParticleCount_;) {
      const std::size_t count =
          std::min(kFrameChunkParticles, sampleParticleCount_ - offset);
      const auto blocks = static_cast<unsigned int>(
          (count + kCaptureThreads - 1) / kCaptureThreads);
      gatherParticleTypeSamples<<<blocks, kCaptureThreads>>>(
          particles_.type, particles_.count, sampleParticleCount_, offset,
          count, deviceSampledTypes);
      checkCuda(cudaGetLastError(),
                "Could not launch the particle type sampling kernel");
      checkCuda(cudaMemcpy(sampledTypes_.data() + offset, deviceSampledTypes,
                           count * sizeof(std::uint8_t),
                           cudaMemcpyDeviceToHost),
                "Could not copy sampled particle types");
      offset += count;
    }
    (void)cudaFree(deviceSampledTypes);
    deviceSampledTypes = nullptr;
    if (!std::all_of(sampledTypes_.begin(), sampledTypes_.end(),
                     [](std::uint8_t type) {
                       return isValidParticleType(type);
                     })) {
      throw std::runtime_error(
          "Cannot capture a frame with invalid particle types");
    }
  } catch (...) {
    if (deviceSampledTypes != nullptr)
      (void)cudaFree(deviceSampledTypes);
    if (devicePositions_ != nullptr)
      (void)cudaFree(devicePositions_);
    if (deviceBlockBounds_ != nullptr)
      (void)cudaFree(deviceBlockBounds_);
    if (deviceInvalidPosition_ != nullptr)
      (void)cudaFree(deviceInvalidPosition_);
    throw;
  }
}

DeviceFrameWriter::~DeviceFrameWriter() noexcept {
  if (devicePositions_ != nullptr)
    (void)cudaFree(devicePositions_);
  if (deviceBlockBounds_ != nullptr)
    (void)cudaFree(deviceBlockBounds_);
  if (deviceInvalidPosition_ != nullptr)
    (void)cudaFree(deviceInvalidPosition_);
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
                         double simulationTime) const {
  PositionBounds bounds;
  bounds.minimum.fill(std::numeric_limits<double>::infinity());
  bounds.maximum.fill(-std::numeric_limits<double>::infinity());

  const std::size_t maximumBlocks =
      (std::min(kFrameChunkParticles, sampleParticleCount_) +
       kCaptureThreads - 1) /
      kCaptureThreads;
  std::vector<double> hostBlockBounds(maximumBlocks * kBoundsValuesPerBlock);

  for (std::size_t offset = 0; offset < sampleParticleCount_;) {
    const std::size_t count =
        std::min(kFrameChunkParticles, sampleParticleCount_ - offset);
    const auto blocks = static_cast<unsigned int>(
        (count + kCaptureThreads - 1) / kCaptureThreads);
    checkCuda(cudaMemset(deviceInvalidPosition_, 0, sizeof(int)),
              "Could not reset the sampling validation flag");
    gatherPositionBounds<<<blocks, kCaptureThreads,
                           kCaptureThreads * kBoundsValuesPerBlock *
                               sizeof(double)>>>(
        particles_.x, particles_.y, particles_.z, particles_.count,
        sampleParticleCount_, offset, count, deviceBlockBounds_,
        deviceInvalidPosition_);
    checkCuda(cudaGetLastError(),
              "Could not launch the graphical bounds kernel");

    int invalidPosition = 0;
    checkCuda(cudaMemcpy(&invalidPosition, deviceInvalidPosition_, sizeof(int),
                         cudaMemcpyDeviceToHost),
              "Could not validate graphical samples");
    if (invalidPosition != 0)
      throw std::runtime_error(
          "Cannot quantize a graphical frame with non-finite positions");

    checkCuda(cudaMemcpy(hostBlockBounds.data(), deviceBlockBounds_,
                         blocks * kBoundsValuesPerBlock * sizeof(double),
                         cudaMemcpyDeviceToHost),
              "Could not copy graphical bounds");
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t base = block * kBoundsValuesPerBlock;
      for (std::size_t component = 0; component < kPositionComponents;
           ++component) {
        bounds.minimum[component] =
            std::min(bounds.minimum[component],
                     hostBlockBounds[base + component]);
        bounds.maximum[component] =
            std::max(bounds.maximum[component],
                     hostBlockBounds[base + kPositionComponents + component]);
      }
    }
    offset += count;
  }

  const QuantizedPositionChunkReader reader =
      [this, bounds](std::size_t offset, std::size_t count,
                     std::uint16_t *xyz) {
        const auto blocks = static_cast<unsigned int>(
            (count + kCaptureThreads - 1) / kCaptureThreads);
        gatherQuantizedPositionSamples<<<blocks, kCaptureThreads>>>(
            particles_.x, particles_.y, particles_.z, particles_.count,
            sampleParticleCount_, offset, count, bounds.minimum[0],
            bounds.minimum[1], bounds.minimum[2], bounds.maximum[0],
            bounds.maximum[1], bounds.maximum[2], devicePositions_);
        checkCuda(cudaGetLastError(),
                  "Could not launch the graphical quantization kernel");
        checkCuda(cudaMemcpy(xyz, devicePositions_,
                             count * kPositionComponents *
                                 sizeof(std::uint16_t),
                             cudaMemcpyDeviceToHost),
                  "Could not copy quantized samples to the frame writer");
      };

  const ParticleTypeChunkReader typeReader =
      [this](std::size_t offset, std::size_t count, std::uint8_t *types) {
        std::copy_n(sampledTypes_.data() + offset, count, types);
      };

  return spool.writeQuantizedPositions(
      sequence, simulationStep, simulationTime, sampleParticleCount_, bounds,
      reader, particles_.count, typeReader);
}

} // namespace nbody::streaming
