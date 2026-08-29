#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <thrust/device_vector.h>

#include "octree_builder.hpp"

namespace 
{
    void checkCuda(cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }

    const char* constructionErrorMessage(int errorCode)
    {
        switch (errorCode)
        {
            case 1:
                return "Invalid radix edge metadata";

            case 2:
                return "Invalid radix ancestor while constructing octree";

            case 3:
                return "Could not find the correct octree parent level";

            case 4:
                return "Two octree nodes attempted to occupy the same parent octant";

            case 5:
                return "A radix leaf terminates at an invalid octree level";

            case 6:
                return "Generated octree node ID is outside the allocated range";

            case 7:
                return "Invalid representative Morton key";

            default:
                return "Unknown sparse-octree construction error";
        }
    }
} // namespace

__device__ std::uint32_t mortonPrefixAtLevel(std::uint32_t key, int level)
{
    if (level == 0)
        return 0u;

    const int remainingBits = kMortonSpatialBits - level * kMortonDimensions;

    const std::uint32_t mask = kMortonSpatialMask & (~0u << remainingBits);

    return key & mask;
}

__device__ int octantAtLevel(std::uint32_t key, int level)
{
    const int shift = kMortonSpatialBits - level * kMortonDimensions;

    return static_cast<int>((key >> shift) & 0x7u);
}

__device__ bool representativeKeyForRadixNode(const ConstTreeDeviceView& radixTree, const ConstMortonLeafGroupsDeviceView& groups, int radixNode, std::uint32_t& key)
{
    if (radixNode < radixTree.nLeaves)
    {
        key = groups.uniqueKeys[radixNode];
        return true;
    }

    const int internalIndex = radixNode - radixTree.nLeaves;
    const int first = radixTree.rangeFirst[internalIndex];

    if (first < 0 || first >= radixTree.nLeaves)
        return false;

    key = groups.uniqueKeys[first];

    return true;
}

__global__ void materializeSparseOctreeKernel(OctreeDeviceView octree, ConstTreeDeviceView radixTree, ConstMortonLeafGroupsDeviceView groups, ConstRadixToOctreePlanDeviceView plan, int* errorCode)
{
    const int radixNode = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (radixNode >= plan.nRadixNodes)
        return;

    const int count = plan.edgeNodeCount[radixNode];
    const int nodeLevel = plan.radixLevel[radixNode];

    if (count < 0 || nodeLevel < 0 || nodeLevel > kOctreeMaxLevel)
    {
        atomicCAS(errorCode, 0, 1);
        return;
    }

    //A radix node can exist without introducing a new octree node.
    if (count == 0)
    {
        if (radixNode < radixTree.nLeaves)
            atomicCAS(errorCode, 0, 5);

        return;
    }

    const int parentLevel = nodeLevel - count;

    if (parentLevel < 0)
    {
        atomicCAS(errorCode, 0, 1);
        return;
    }

    std::uint32_t key = 0u;

    if (!representativeKeyForRadixNode(radixTree, groups, radixNode, key))
    {
        atomicCAS(errorCode, 0, 7);
        return;
    }

    //Find the octree node from which this radix edge starts.
    //Some radix ancestors can have edgeNodeCount == 0 because binary-prefix levels do not necessarily coincide with 3-bit octree levels.
    int parentOctreeNode = 0;
    int ancestor =radixTree.parent[radixNode];
    int iterations = 0;

    while (ancestor != -1)
    {
        if (ancestor < 0 || ancestor >= plan.nRadixNodes)
        {
            atomicCAS(errorCode, 0, 2);
            return;
        }

        const int ancestorCount = plan.edgeNodeCount[ancestor];

        if (ancestorCount > 0)
        {
            const int ancestorLevel = plan.radixLevel[ancestor];
            if (ancestorLevel != parentLevel)
            {
                atomicCAS(errorCode, 0, 3);
                return;
            }

            parentOctreeNode = 1 + plan.edgeNodeOffset[ancestor] + ancestorCount - 1;
            break;
        }

        ancestor = radixTree.parent[ancestor];
        ++iterations;

        if (iterations > plan.nRadixNodes)
        {
            atomicCAS(errorCode, 0, 2);
            return;
        }
    }

    // If no radix ancestor generated an octree node, this edge must start directly from the explicit octree root at level zero
    if (ancestor == -1 && parentLevel != 0)
    {
        atomicCAS(errorCode, 0, 3);
        return;
    }

    const int firstNode = 1 + plan.edgeNodeOffset[radixNode];

    for (int local = 0; local < count; ++local)
    {
        const int node = firstNode + local;

        if (node <= 0 || node >= octree.nNodes)
        {
            atomicCAS(errorCode, 0, 6);
            return;
        }

        const int level = parentLevel + local + 1;
        const int parent = (local == 0) ? parentOctreeNode : node - 1;
        octree.level[node] = level;
        octree.prefix[node] = mortonPrefixAtLevel(key, level);
        octree.parent[node] = parent;
        const int octant =octantAtLevel(key, level);

        //atomicCAS is useful because it makes simultaneous writes safe and it detects a topology bug if two nodes try to occupy the same octant
        const int previous = atomicCAS(&octree.children[parent * 8 + octant], -1, node);

        if (previous != -1 && previous != node)
        {
            atomicCAS(errorCode, 0, 4);
            return;
        }
    }

    /*
    * A radix leaf represents one occupied Morton group.
    *
    * The corresponding octree leaf is materialized at the
    * shallowest level that still separates this group from
    * the other occupied groups.
    */
    if (radixNode < radixTree.nLeaves)
    {
        const int leaf = firstNode + count - 1;
        const int leafLevel = octree.level[leaf];

        if (leafLevel <= 0 || leafLevel > kOctreeMaxLevel)
        {
            atomicCAS(errorCode, 0, 5);
            return;
        }

        octree.firstParticle[leaf] = groups.firstParticle[radixNode];
        octree.particleCount[leaf] = groups.particleCount[radixNode];
    }
}

void allocateOctreeTopology(Octree& octree, int nNodes)
{
    if (nNodes <= 0)
        throw std::invalid_argument("Octree node count must be positive");

    if (!octree.children.empty() || !octree.parent.empty() ||
        !octree.level.empty() || !octree.prefix.empty() ||
        !octree.firstParticle.empty() || !octree.particleCount.empty() ||
        !octree.mass.empty() || !octree.comX.empty() || !octree.comY.empty() ||
        !octree.comZ.empty() || !octree.centerX.empty() ||
        !octree.centerY.empty() || !octree.centerZ.empty() ||
        !octree.halfSize.empty() || !octree.pendingChildren.empty() ||
        !octree.childCount.empty())
        throw std::logic_error("Octree memory is already allocated");

    const std::size_t nodeCount = static_cast<std::size_t>(nNodes);
    Octree staged;
    staged.children.resize(nodeCount * 8);
    staged.parent.resize(nodeCount);
    staged.level.resize(nodeCount);
    staged.prefix.resize(nodeCount);
    staged.firstParticle.resize(nodeCount);
    staged.particleCount.resize(nodeCount);
    staged.nNodes = nNodes;
    octree = std::move(staged);
}

void buildSparseOctreeTopology(Octree& octree, const Tree& radixTree, const MortonLeafGroups& groups, const RadixToOctreePlan& plan)
{
    if (groups.nParticles <= 0 || groups.nGroups <= 0)
        throw std::invalid_argument("Morton groups are empty");

    if (groups.nGroups !=radixTree.nLeaves)
        throw std::invalid_argument("Morton groups and radix tree have different leaf counts");

    if (groups.uniqueKeys.empty() || groups.firstParticle.empty() ||
        groups.particleCount.empty())
        throw std::logic_error("Morton group arrays are not allocated");

    const int expectedRadixNodes = 2 * radixTree.nLeaves - 1;

    if (plan.nRadixNodes != expectedRadixNodes)
        throw std::invalid_argument("Radix-to-octree plan does not match the radix tree");

    if (plan.nOctreeNodes <= 0)
        throw std::invalid_argument("Radix-to-octree plan has no octree nodes");

    if (octree.nNodes != plan.nOctreeNodes)
        throw std::invalid_argument("Allocated octree size does not match the construction plan");

    if (octree.children.empty() || octree.parent.empty() ||
        octree.level.empty() || octree.prefix.empty() ||
        octree.firstParticle.empty() || octree.particleCount.empty())
        throw std::logic_error("Octree topology memory is not allocated");

    if (radixTree.parent.empty())
        throw std::logic_error("Radix parent array is not allocated");

    if (radixTree.nLeaves > 1 && (radixTree.rangeFirst.empty() || radixTree.prefixLength.empty()))
        throw std::logic_error("Radix metadata is incomplete");

    if (plan.radixLevel.empty() || plan.edgeNodeCount.empty() || plan.edgeNodeOffset.empty())
        throw std::logic_error("Radix-to-octree plan arrays are not allocated");

    OctreeDeviceView octreeView = octree.device_view();

    //Empty child slots and parent IDs use -1.
    checkCuda(cudaMemset(octreeView.children, 0xFF, static_cast<std::size_t>(octree.nNodes) * 8 * sizeof(int)), "initialize octree.children");
    checkCuda(cudaMemset(octreeView.parent, 0xFF, static_cast<std::size_t>(octree.nNodes) * sizeof(int)), "initialize octree.parent");
    checkCuda(
        cudaMemset(
            octreeView.level,
            0xFF,
            static_cast<std::size_t>(
                octree.nNodes) *
                sizeof(int)),
        "initialize octree.level");
    checkCuda(
        cudaMemset(
            octreeView.prefix,
            0,
            static_cast<std::size_t>(
                octree.nNodes) *
                sizeof(std::uint32_t)),
        "initialize octree.prefix");
    checkCuda(
        cudaMemset(
            octreeView.firstParticle,
            0xFF,
            static_cast<std::size_t>(
                octree.nNodes) *
                sizeof(int)),
        "initialize octree.firstParticle");
    checkCuda(cudaMemset(octreeView.particleCount, 0, static_cast<std::size_t>(octree.nNodes) * sizeof(int)), "initialize octree.particleCount");

    const int rootLevel = 0;
    const std::uint32_t rootPrefix = 0u;

    checkCuda(cudaMemcpy(octreeView.level, &rootLevel, sizeof(int), cudaMemcpyHostToDevice), "initialize octree root level");
    checkCuda(cudaMemcpy(octreeView.prefix, &rootPrefix, sizeof(std::uint32_t), cudaMemcpyHostToDevice), "initialize octree root prefix");

    thrust::device_vector<int> errorCodeDevice(1, 0);
    int* const errorCode = nbody::deviceData(errorCodeDevice);

    constexpr int threadsPerBlock = 256;
    const int blocks = (plan.nRadixNodes + threadsPerBlock - 1) / threadsPerBlock;

    materializeSparseOctreeKernel<<<blocks, threadsPerBlock>>>(octreeView, radixTree.device_view(), groups.device_view(), plan.device_view(), errorCode);

    checkCuda(cudaGetLastError(), "materializeSparseOctreeKernel launch");
    checkCuda(cudaDeviceSynchronize(), "materializeSparseOctreeKernel execution");

    int hostErrorCode = 0;

    checkCuda(cudaMemcpy(&hostErrorCode, errorCode, sizeof(int), cudaMemcpyDeviceToHost), "copy octree construction error code");

    if (hostErrorCode != 0)
        throw std::runtime_error(constructionErrorMessage(hostErrorCode));

    octree.nParticles = groups.nParticles;
    octree.nLeaves = groups.nGroups;
    octree.root = 0;
}

void freeOctree(Octree& octree)
{
    octree = {};
}
