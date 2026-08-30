#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

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

__device__ bool representativeKeyForRadixNode(const ConstTreeView& radixTree, const MortonLeafGroupsView& groups, int radixNode, std::uint32_t& key)
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

__global__ void materializeSparseOctreeKernel(OctreeView octree, ConstTreeView radixTree, MortonLeafGroupsView groups, RadixToOctreePlanView plan, int* errorCode)
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

    if (!octree.children.empty() || !octree.parent.empty() || !octree.level.empty() || !octree.prefix.empty() || !octree.firstParticle.empty() || !octree.particleCount.empty() || !octree.mass.empty() || !octree.comX.empty() || !octree.comY.empty() || !octree.comZ.empty() || !octree.centerX.empty() || !octree.centerY.empty() || !octree.centerZ.empty() || !octree.halfSize.empty() || !octree.pendingChildren.empty() || !octree.childCount.empty())
        throw std::logic_error("Octree memory is already allocated");
    
    const std::size_t nodeCount = static_cast<std::size_t>(nNodes);

    try
    {
        octree.children.allocate(nodeCount * 8);
        octree.parent.allocate(nodeCount);
        octree.level.allocate(nodeCount);
        octree.prefix.allocate(nodeCount);
        octree.firstParticle.allocate(nodeCount);
        octree.particleCount.allocate(nodeCount);
    }
    catch (...)
    {
        freeOctree(octree);
        throw;
    }

    octree.nNodes = nNodes;
}

void buildSparseOctreeTopology(Octree& octree, const Tree& radixTree, const MortonLeafGroups& groups, const RadixToOctreePlan& plan)
{
    if (groups.nParticles <= 0 || groups.nGroups <= 0)
        throw std::invalid_argument("Morton groups are empty");

    if (groups.nGroups !=radixTree.nLeaves)
        throw std::invalid_argument("Morton groups and radix tree have different leaf counts");

    if (groups.uniqueKeys.data() == nullptr || groups.firstParticle.data() == nullptr || groups.particleCount.data() == nullptr)
        throw std::logic_error("Morton group arrays are not allocated");

    const int expectedRadixNodes = 2 * radixTree.nLeaves - 1;

    if (plan.nRadixNodes != expectedRadixNodes)
        throw std::invalid_argument("Radix-to-octree plan does not match the radix tree");

    if (plan.nOctreeNodes <= 0)
        throw std::invalid_argument("Radix-to-octree plan has no octree nodes");

    if (octree.nNodes != plan.nOctreeNodes)
        throw std::invalid_argument("Allocated octree size does not match the construction plan");

    if (octree.children.data() == nullptr || octree.parent.data() == nullptr || octree.level.data() == nullptr || octree.prefix.data() == nullptr || octree.firstParticle.data() == nullptr || octree.particleCount.data() == nullptr)
        throw std::logic_error("Octree topology memory is not allocated");

    if (radixTree.parent.data() == nullptr)
        throw std::logic_error("Radix parent array is not allocated");

    if (radixTree.nLeaves > 1 && (radixTree.rangeFirst.data() == nullptr || radixTree.prefixLength.data() == nullptr))
        throw std::logic_error("Radix metadata is incomplete");

    if (plan.radixLevel.data() == nullptr || plan.edgeNodeCount.data() == nullptr || plan.edgeNodeOffset.data() == nullptr)
        throw std::logic_error("Radix-to-octree plan arrays are not allocated");

    //Empty child slots and parent IDs use -1.
    checkCuda(cudaMemset(octree.children.data(), 0xFF, static_cast<std::size_t>(octree.nNodes) * 8 * sizeof(int)), "initialize octree.children");
    checkCuda(cudaMemset(octree.parent.data(), 0xFF, static_cast<std::size_t>(octree.nNodes) * sizeof(int)), "initialize octree.parent");
    checkCuda(
        cudaMemset(
            octree.level.data(),
            0xFF,
            static_cast<std::size_t>(
                octree.nNodes) *
                sizeof(int)),
        "initialize octree.level");
    checkCuda(
        cudaMemset(
            octree.prefix.data(),
            0,
            static_cast<std::size_t>(
                octree.nNodes) *
                sizeof(std::uint32_t)),
        "initialize octree.prefix");
    checkCuda(
        cudaMemset(
            octree.firstParticle.data(),
            0xFF,
            static_cast<std::size_t>(
                octree.nNodes) *
                sizeof(int)),
        "initialize octree.firstParticle");
    checkCuda(cudaMemset(octree.particleCount.data(), 0, static_cast<std::size_t>(octree.nNodes) * sizeof(int)), "initialize octree.particleCount");

    const int rootLevel = 0;
    const std::uint32_t rootPrefix = 0u;

    checkCuda(cudaMemcpy(octree.level.data(), &rootLevel, sizeof(int), cudaMemcpyHostToDevice), "initialize octree root level");
    checkCuda(cudaMemcpy(octree.prefix.data(), &rootPrefix, sizeof(std::uint32_t), cudaMemcpyHostToDevice), "initialize octree root prefix");

    int* errorCodeDevice = nullptr;

    try
    {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&errorCodeDevice), sizeof(int)), "allocate octree construction error code");
        checkCuda(cudaMemset(errorCodeDevice, 0, sizeof(int)), "initialize octree construction error code");

        constexpr int threadsPerBlock = 256;
        const int blocks = (plan.nRadixNodes + threadsPerBlock - 1) / threadsPerBlock;

        const MortonLeafGroupsView groupsView = groups.view();
        const RadixToOctreePlanView planView = plan.view();
        const ConstTreeView radixTreeView = radixTree.view();
        const OctreeView octreeView = octree.view();
       
        materializeSparseOctreeKernel<<<blocks, threadsPerBlock>>>(octreeView, radixTreeView, groupsView, planView, errorCodeDevice);

        checkCuda(cudaGetLastError(), "materializeSparseOctreeKernel launch");
        checkCuda(cudaDeviceSynchronize(), "materializeSparseOctreeKernel execution");

        int errorCode = 0;

        checkCuda(cudaMemcpy(&errorCode, errorCodeDevice, sizeof(int), cudaMemcpyDeviceToHost), "copy octree construction error code");

        if (errorCode != 0)
            throw std::runtime_error(constructionErrorMessage(errorCode));

        cudaFree(errorCodeDevice);
        errorCodeDevice = nullptr;
    }
    catch (...)
    {
        cudaFree(errorCodeDevice);
        throw;
    }

    octree.nParticles = groups.nParticles;
    octree.nLeaves = groups.nGroups;
    octree.root = 0;
}

void freeOctree(Octree& octree) noexcept
{
    octree = Octree{};
}