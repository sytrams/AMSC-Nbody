#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

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

    void ensureTemporaryStorage(RadixToOctreePlan& plan, std::size_t requiredBytes)
    {
        if (requiredBytes <= plan.temporaryStorageBytes)
            return;

        void* newStorage = nullptr;

        checkCuda(cudaMalloc(&newStorage, requiredBytes), "cudaMalloc radix-to-octree temporary storage");

        cudaFree(plan.temporaryStorage);

        plan.temporaryStorage = newStorage;
        plan.temporaryStorageBytes = requiredBytes;
    }

    const char* planErrorMessage(int errorCode)
    {
        switch (errorCode)
        {
            case 1:
                return "A Morton key uses bits outside the configured 30-bit spatial layout";

            case 2:
                return "A radix node has an invalid parent";

            case 3:
                return "A radix child has an octree level smaller than its parent";

            case 4:
                return "A radix internal node has an invalid prefix length";

            default:
                return "Unknown radix-to-octree planning error";
        }
    }
} // namespace

__device__ int spatialPrefixLength(const Tree& radixTree, int radixNode)
{
    if (radixNode < radixTree.nLeaves)
        return kMortonSpatialBits;  // A radix leaf represents one complete Morton key

    const int internalIndex = radixNode - radixTree.nLeaves;
    const int rawPrefix = radixTree.prefixLength[internalIndex];
    return rawPrefix - kMortonPaddingBits;
}

__device__ int octreeLevelForRadixNode(const Tree& radixTree,int radixNode)
{
    const int prefix = spatialPrefixLength(radixTree, radixNode);
    return prefix / kMortonDimensions;
}

__global__ void computeRadixToOctreePlanKernel(Tree radixTree, const std::uint32_t* uniqueKeys, int nRadixNodes, int* radixLevel, int* edgeNodeCount, int* errorCode)
{
    const int radixNode =static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (radixNode >= nRadixNodes)
        return;

    if (radixNode < radixTree.nLeaves)
    {
        const std::uint32_t key = uniqueKeys[radixNode];

        if ((key & ~kMortonSpatialMask) != 0u)
        {
            atomicCAS(errorCode,0, 1);
            radixLevel[radixNode] = 0;
            edgeNodeCount[radixNode] = 0;
            return;
        }
    }
    else
    {
        const int internalIndex = radixNode - radixTree.nLeaves;
        const int rawPrefix =radixTree.prefixLength[internalIndex];
        const int spatialPrefix =rawPrefix - kMortonPaddingBits;

        if (spatialPrefix < 0 || spatialPrefix >= kMortonSpatialBits)
        {
            atomicCAS(errorCode, 0, 4);
            radixLevel[radixNode] = 0;
            edgeNodeCount[radixNode] = 0;
            return;
        }
    }

    const int parentNode = radixTree.parent[radixNode];
    int parentLevel = 0;

    if (parentNode != -1)
    {
        if (parentNode < radixTree.nLeaves || parentNode >= nRadixNodes)
        {
            atomicCAS(errorCode, 0, 2);
            radixLevel[radixNode] = 0;
            edgeNodeCount[radixNode] = 0;
            return;
        }

        parentLevel = octreeLevelForRadixNode(radixTree, parentNode);
    }

    int nodeLevel = 0;

    if (radixNode < radixTree.nLeaves)
        nodeLevel = parentLevel + 1;
    else
        nodeLevel = octreeLevelForRadixNode(radixTree, radixNode);

    if (nodeLevel < 0 || nodeLevel > kOctreeMaxLevel)
    {
        atomicCAS(errorCode, 0, 3);
        radixLevel[radixNode] = 0;
        edgeNodeCount[radixNode] = 0;
        return;
    }

    if (nodeLevel < parentLevel)
    {
        atomicCAS(errorCode, 0, 3);
        radixLevel[radixNode] = 0;
        edgeNodeCount[radixNode] = 0;
        return;
    }

    radixLevel[radixNode] = nodeLevel;
    edgeNodeCount[radixNode] = nodeLevel - parentLevel;
}

void allocateRadixToOctreePlan(RadixToOctreePlan& plan, int nRadixNodes)
{
    if (nRadixNodes <= 0)
        throw std::invalid_argument("nRadixNodes must be positive");

    if (plan.radixLevel != nullptr || plan.edgeNodeCount != nullptr || plan.edgeNodeOffset != nullptr || plan.errorCodeDevice != nullptr || plan.temporaryStorage != nullptr)
        throw std::logic_error("RadixToOctreePlan memory is already allocated");

    const std::size_t count = static_cast<std::size_t>(nRadixNodes);

    try
    {
        allocateDeviceArray(plan.radixLevel, count, "plan.radixLevel");
        allocateDeviceArray(plan.edgeNodeCount, count, "plan.edgeNodeCount");
        allocateDeviceArray(plan.edgeNodeOffset, count, "plan.edgeNodeOffset");
        allocateDeviceArray(plan.errorCodeDevice, 1, "plan.errorCodeDevice");
    }
    catch (...)
    {
        freeRadixToOctreePlan(plan);
        throw;
    }

    plan.nRadixNodes =nRadixNodes;
    plan.nOctreeNodes = 0;
}

void buildRadixToOctreePlan(RadixToOctreePlan& plan, const Tree& radixTree, const MortonLeafGroups& groups)
{
    if (radixTree.nLeaves <= 0)
        throw std::invalid_argument("The radix tree has no leaves");

    if (groups.nGroups != radixTree.nLeaves)
        throw std::invalid_argument("Morton group count does not match radix-tree leaf count");

    if (groups.uniqueKeys.data() == nullptr)
        throw std::logic_error("Morton unique keys are not allocated");

    if (radixTree.parent == nullptr)
        throw std::logic_error("Radix-tree parent array is not allocated");

    if (radixTree.nLeaves > 1 && radixTree.prefixLength == nullptr)
        throw std::logic_error("Radix-tree prefix metadata is not allocated");

    const int expectedRadixNodes = 2 * radixTree.nLeaves - 1;

    if (plan.nRadixNodes != expectedRadixNodes)
        throw std::invalid_argument("RadixToOctreePlan size does not match the radix tree");

    if (plan.radixLevel == nullptr || plan.edgeNodeCount == nullptr || plan.edgeNodeOffset == nullptr || plan.errorCodeDevice == nullptr)
        throw std::logic_error("RadixToOctreePlan memory is not allocated");

    plan.nOctreeNodes = 0;

    checkCuda(cudaMemset(plan.errorCodeDevice, 0, sizeof(int)), "cudaMemset plan.errorCodeDevice");

    constexpr int threadsPerBlock = 256;

    const int blocks =(plan.nRadixNodes + threadsPerBlock - 1) / threadsPerBlock;

    computeRadixToOctreePlanKernel<<<blocks, threadsPerBlock>>>(radixTree, groups.uniqueKeys.data(), plan.nRadixNodes, plan.radixLevel, plan.edgeNodeCount, plan.errorCodeDevice);

    checkCuda(cudaGetLastError(), "computeRadixToOctreePlanKernel launch");
    checkCuda(cudaDeviceSynchronize(), "computeRadixToOctreePlanKernel execution");

    int errorCode = 0;

    checkCuda(cudaMemcpy(&errorCode, plan.errorCodeDevice, sizeof(int), cudaMemcpyDeviceToHost), "copy radix-to-octree error code");

    if (errorCode != 0)
        throw std::runtime_error(planErrorMessage(errorCode));

    std::size_t scanStorageBytes = 0;

    checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, scanStorageBytes, plan.edgeNodeCount, plan.edgeNodeOffset, plan.nRadixNodes), "DeviceScan radix-edge storage query");

    ensureTemporaryStorage(plan, scanStorageBytes);

    checkCuda(cub::DeviceScan::ExclusiveSum(plan.temporaryStorage, scanStorageBytes, plan.edgeNodeCount, plan.edgeNodeOffset, plan.nRadixNodes), "DeviceScan radix-edge execution");

    const int lastIndex = plan.nRadixNodes - 1;

    int lastOffset = 0;
    int lastCount = 0;

    checkCuda(cudaMemcpy(&lastOffset, plan.edgeNodeOffset + lastIndex, sizeof(int), cudaMemcpyDeviceToHost), "copy final octree offset");

    checkCuda(cudaMemcpy(&lastCount, plan.edgeNodeCount + lastIndex, sizeof(int), cudaMemcpyDeviceToHost), "copy final octree count");

    const long long generatedNodes = static_cast<long long>(lastOffset) + static_cast<long long>(lastCount);

    if (generatedNodes < 0 || generatedNodes >= std::numeric_limits<int>::max())
        throw std::overflow_error("The computed octree node count does not fit in int");

    plan.nOctreeNodes = 1 + static_cast<int>(generatedNodes);
}

void freeRadixToOctreePlan(RadixToOctreePlan& plan)
{
    cudaFree(plan.radixLevel);
    cudaFree(plan.edgeNodeCount);
    cudaFree(plan.edgeNodeOffset);
    cudaFree(plan.errorCodeDevice);
    cudaFree(plan.temporaryStorage);

    plan.radixLevel = nullptr;
    plan.edgeNodeCount = nullptr;
    plan.edgeNodeOffset = nullptr;
    plan.errorCodeDevice = nullptr;
    plan.temporaryStorage = nullptr;

    plan.nRadixNodes = 0;
    plan.nOctreeNodes = 0;
    plan.temporaryStorageBytes = 0;
}