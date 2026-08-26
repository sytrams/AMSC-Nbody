#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "octree_geometry.hpp"
#include "radix_to_octree.hpp"

namespace
{

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
}

template <typename T>
void allocateDeviceArray(T*& pointer, std::size_t count, const char* name)
{
    if (count == 0)
        return;

    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T));

    if (error != cudaSuccess)
        throw std::runtime_error(std::string("cudaMalloc ") + name + " failed: " + cudaGetErrorString(error));
}

const char* geometryErrorMessage(int errorCode)
{
    switch (errorCode)
    {
        case 1:
            return "Octree node has an invalid level";

        case 2:
            return "Octree node has an invalid Morton prefix";

        default:
            return "Unknown octree geometry error";
    }
}

} // namespace


__device__ bool prefixIsValidForLevel(std::uint32_t prefix, int level)
{
    // Morton keys use only the lowest 30 bits of the uint32_t.
    if ((prefix & ~kMortonSpatialMask) != 0u)
        return false;

    if (level == 0)
        return prefix == 0u;

    const int remainingBits = kMortonSpatialBits - level * kMortonDimensions;
    const std::uint32_t mask = kMortonSpatialMask & (~0u << remainingBits);

    // Bits below the prefix represented by this octree level must be zero.
    return (prefix & ~mask) == 0u;
}


__global__ void computeOctreeGeometryKernel(Octree octree, double rootCenterX, double rootCenterY, double rootCenterZ, double rootSide, int* errorCode)
{
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (node >= octree.nNodes)
        return;

    const int level = octree.level[node];

    if (level < 0 || level > kOctreeMaxLevel)
    {
        atomicCAS(errorCode, 0, 1);
        return;
    }

    const std::uint32_t prefix = octree.prefix[node];

    if (!prefixIsValidForLevel(prefix, level))
    {
        atomicCAS(errorCode, 0, 2);
        return;
    }

    // Root geometry.
    double centerX = rootCenterX;
    double centerY = rootCenterY;
    double centerZ = rootCenterZ;
    double halfSize = rootSide * 0.5;

    // Reconstruct the path encoded in the Morton prefix.
    for (int depth = 1; depth <= level; ++depth)
    {
        const int shift = kMortonSpatialBits - depth * kMortonDimensions;
        const int octant = static_cast<int>((prefix >> shift) & 0x7u);
        const double childHalfSize = halfSize * 0.5;

        centerX +=(octant & 0x4) ? childHalfSize : -childHalfSize;
        centerY += (octant & 0x2) ? childHalfSize : -childHalfSize;
        centerZ += (octant & 0x1) ? childHalfSize : -childHalfSize;
        halfSize = childHalfSize;
    }

    octree.centerX[node] = centerX;
    octree.centerY[node] = centerY;
    octree.centerZ[node] = centerZ;
    octree.halfSize[node] = halfSize;
}


void allocateOctreeGeometry(Octree& octree)
{
    if (octree.nNodes <= 0)
        throw std::invalid_argument("Octree topology must be allocated before geometry");

    if (octree.level == nullptr || octree.prefix == nullptr)
        throw std::logic_error("Octree topology arrays required for geometry are not allocated");

    if (octree.centerX != nullptr || octree.centerY != nullptr || octree.centerZ != nullptr || octree.halfSize != nullptr)
        throw std::logic_error("Octree geometry is already allocated");

    const std::size_t count = static_cast<std::size_t>(octree.nNodes);

    try
    {
        allocateDeviceArray(octree.centerX, count, "octree.centerX");
        allocateDeviceArray(octree.centerY, count, "octree.centerY");
        allocateDeviceArray(octree.centerZ, count, "octree.centerZ");
        allocateDeviceArray(octree.halfSize, count, "octree.halfSize");
    }
    catch (...)
    {
        cudaFree(octree.centerX);
        cudaFree(octree.centerY);
        cudaFree(octree.centerZ);
        cudaFree(octree.halfSize);

        octree.centerX = nullptr;
        octree.centerY = nullptr;
        octree.centerZ = nullptr;
        octree.halfSize = nullptr;

        throw;
    }
}


void computeOctreeGeometry(Octree& octree, const Bbox& boundingBox)
{
    if (octree.nNodes <= 0 || octree.root != 0)
        throw std::logic_error("Octree topology is not valid");

    if (octree.level == nullptr || octree.prefix == nullptr)
        throw std::logic_error("Octree topology arrays required for geometry are not allocated");
    
    if (octree.centerX == nullptr || octree.centerY == nullptr || octree.centerZ == nullptr ||  octree.halfSize == nullptr)
        throw std::logic_error("Octree geometry arrays are not allocated");
    
        // Copy the four bounding-box values to the host once.
    const BboxValues bounds = boundingBox.values();

    if (!std::isfinite(bounds.center_x) || !std::isfinite(bounds.center_y) || !std::isfinite(bounds.center_z) || !std::isfinite(bounds.side) || bounds.side <= 0.0)
        throw std::invalid_argument("Bounding box geometry is not valid");

    int* errorCodeDevice = nullptr;

    try
    {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&errorCodeDevice), sizeof(int)), "allocate octree geometry error code");
        checkCuda(cudaMemset(errorCodeDevice, 0, sizeof(int)), "initialize octree geometry error code");

        constexpr int threadsPerBlock = 256;
        const int blocks =(octree.nNodes + threadsPerBlock - 1) / threadsPerBlock;

        computeOctreeGeometryKernel<<<blocks, threadsPerBlock>>>(octree, bounds.center_x, bounds.center_y, bounds.center_z, bounds.side, errorCodeDevice);

        checkCuda(cudaGetLastError(), "computeOctreeGeometryKernel launch");
        checkCuda(cudaDeviceSynchronize(), "computeOctreeGeometryKernel execution");

        int errorCode = 0;

        checkCuda(cudaMemcpy(&errorCode, errorCodeDevice, sizeof(int), cudaMemcpyDeviceToHost), "copy octree geometry error code");

        if (errorCode != 0)
            throw std::runtime_error(geometryErrorMessage(errorCode));
        
        cudaFree(errorCodeDevice);
        errorCodeDevice = nullptr;
    }
    catch (...)
    {
        cudaFree(errorCodeDevice);
        throw;
    }
}