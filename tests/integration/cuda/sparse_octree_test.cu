#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_test.hpp"
#include "morton_leaf_groups.hpp"
#include "octree_builder.hpp"
#include "radix_to_octree.hpp"
#include "tree_builder.hpp"

namespace {

static_assert(
    !std::is_copy_constructible_v<Octree>);

static_assert(
    !std::is_copy_assignable_v<Octree>);

static_assert(
    std::is_move_constructible_v<Octree>);

static_assert(
    std::is_move_assignable_v<Octree>);

void checkCuda(
    cudaError_t error,
    const char* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) +
            " failed: " +
            cudaGetErrorString(error));
    }
}

template <typename T>
class DeviceArray
{
public:
    explicit DeviceArray(
        const std::vector<T>& host)
        : size_(host.size())
    {
        if (size_ == 0)
            return;

        checkCuda(
            cudaMalloc(
                reinterpret_cast<void**>(&data_),
                size_ * sizeof(T)),
            "cudaMalloc DeviceArray");

        try
        {
            checkCuda(
                cudaMemcpy(
                    data_,
                    host.data(),
                    size_ * sizeof(T),
                    cudaMemcpyHostToDevice),
                "copy DeviceArray");
        }
        catch (...)
        {
            cudaFree(data_);
            data_ = nullptr;
            throw;
        }
    }

    ~DeviceArray()
    {
        cudaFree(data_);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(
        const DeviceArray&) = delete;

    const T* get() const noexcept
    {
        return data_;
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

std::uint32_t prefixAtLevel(
    std::uint32_t key,
    int level)
{
    if (level == 0)
        return 0u;

    const int remaining =
        kMortonSpatialBits -
        level * kMortonDimensions;

    const std::uint32_t mask =
        kMortonSpatialMask &
        (~0u << remaining);

    return key & mask;
}

int octantAtLevel(
    std::uint32_t key,
    int level)
{
    const int shift =
        kMortonSpatialBits -
        level * kMortonDimensions;

    return static_cast<int>(
        (key >> shift) & 0x7u);
}

struct HostOctree
{
    std::vector<int> children;
    std::vector<int> parent;
    std::vector<int> level;
    std::vector<std::uint32_t> prefix;
    std::vector<int> firstParticle;
    std::vector<int> particleCount;
};

HostOctree copyOctree(
    const Octree& octree)
{
    HostOctree host;

    const std::size_t n =
        static_cast<std::size_t>(
            octree.nNodes);

    host.children.resize(n * 8);
    host.parent.resize(n);
    host.level.resize(n);
    host.prefix.resize(n);
    host.firstParticle.resize(n);
    host.particleCount.resize(n);

    checkCuda(
        cudaMemcpy(
            host.children.data(),
            octree.children.data(),
            n * 8 * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree children");

    checkCuda(
        cudaMemcpy(
            host.parent.data(),
            octree.parent.data(),
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree parent");

    checkCuda(
        cudaMemcpy(
            host.level.data(),
            octree.level.data(),
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree levels");

    checkCuda(
        cudaMemcpy(
            host.prefix.data(),
            octree.prefix.data(),
            n * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost),
        "copy octree prefixes");

    checkCuda(
        cudaMemcpy(
            host.firstParticle.data(),
            octree.firstParticle.data(),
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy firstParticle");

    checkCuda(
        cudaMemcpy(
            host.particleCount.data(),
            octree.particleCount.data(),
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy particleCount");

    return host;
}

void validateTopology(
    const HostOctree& tree,
    int nNodes)
{
    ASSERT_GT(nNodes, 0) << "Octree contains no nodes";

    ASSERT_EQ(static_cast<int>(tree.level.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.prefix.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.parent.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.firstParticle.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.particleCount.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.children.size()), nNodes * 8);

    EXPECT_EQ(tree.level[0], 0) << "Root level must be zero";

    EXPECT_EQ(tree.prefix[0], 0u) << "Root prefix must be zero";

    EXPECT_EQ(tree.parent[0], -1) << "Root parent must be -1";

    std::vector<int> incoming(
        static_cast<std::size_t>(nNodes),
        0);

    for (int node = 0;
         node < nNodes;
         ++node)
    {
        EXPECT_TRUE(tree.level[node] >= 0 &&
            tree.level[node] <=
                kOctreeMaxLevel) << "Invalid octree level";

        EXPECT_TRUE(tree.prefix[node] ==
                prefixAtLevel(
                    tree.prefix[node],
                    tree.level[node])) << "Octree prefix contains bits "
            "below its level";

        for (int octant = 0;
             octant < 8;
             ++octant)
        {
            const int child =
                tree.children[
                    node * 8 +
                    octant];

            if (child == -1)
                continue;

            ASSERT_GT(child, 0) << "Child ID outside octree";
            ASSERT_LT(child, nNodes) << "Child ID outside octree";

            EXPECT_EQ(tree.parent[child], node)
                << "Child parent pointer does not point back";

            EXPECT_TRUE(tree.level[child] ==
                    tree.level[node] + 1) << "Octree edge skips a level";

            EXPECT_TRUE(octantAtLevel(
                    tree.prefix[child],
                    tree.level[child]) ==
                    octant) << "Child stored in wrong octant";

            EXPECT_TRUE(prefixAtLevel(
                    tree.prefix[child],
                    tree.level[node]) ==
                    tree.prefix[node]) << "Child prefix does not extend "
                "parent prefix";

            ++incoming[child];
        }
    }

    EXPECT_EQ(incoming[0], 0) << "Root has an incoming edge";

    for (int node = 1;
         node < nNodes;
         ++node)
    {
        EXPECT_EQ(incoming[node], 1)
            << "Non-root node does not have exactly one incoming edge";
    }

    std::vector<int> visited(
        static_cast<std::size_t>(nNodes),
        0);

    int visitedCount = 0;

    std::function<void(int)> visit =
        [&](int node)
    {
        ASSERT_GE(node, 0);
        ASSERT_LT(node, nNodes);
        ASSERT_EQ(visited[node], 0)
            << "Cycle or repeated octree node";

        visited[node] = 1;
        ++visitedCount;

        for (int octant = 0;
             octant < 8;
             ++octant)
        {
            const int child =
                tree.children[
                    node * 8 +
                    octant];

            if (child != -1)
                visit(child);
        }
    };

    visit(0);

    EXPECT_EQ(visitedCount, nNodes)
        << "Some octree nodes are unreachable from the root";
}

void runCase(
    const std::string& name,
    const std::vector<std::uint32_t>& sortedKeys)
{
    ASSERT_FALSE(sortedKeys.empty()) << name + ": empty input";

    ASSERT_TRUE(std::is_sorted(
        sortedKeys.begin(),
        sortedKeys.end()))
        << name + ": keys are not sorted";

    DeviceArray<std::uint32_t>
        deviceKeys(sortedKeys);

    MortonLeafGroups groups{};
    Tree radixTree{};
    RadixToOctreePlan plan{};
    Octree octree{};

    const auto cleanup = [&]()
    {
        freeOctree(octree);
        freeRadixToOctreePlan(plan);
        freeTree(radixTree);
        freeMortonLeafGroups(groups);
    };

    try
    {
        const int nParticles =
            static_cast<int>(
                sortedKeys.size());

        allocateMortonLeafGroups(
            groups,
            nParticles);

        buildMortonLeafGroups(
            groups,
            deviceKeys.get(),
            nParticles);

        allocateTree(
            radixTree,
            groups.nGroups);

        buildTreeFromMortonGroups(
            radixTree,
            groups);

        allocateRadixToOctreePlan(
            plan,
            2 * groups.nGroups - 1);

        buildRadixToOctreePlan(
            plan,
            radixTree,
            groups);

        allocateOctreeTopology(
            octree,
            plan.nOctreeNodes);

        buildSparseOctreeTopology(
            octree,
            radixTree,
            groups,
            plan);

        EXPECT_EQ(octree.root, 0) << name + ": root index is not zero";

        EXPECT_EQ(octree.nNodes, plan.nOctreeNodes) << name + ": wrong node count";

        EXPECT_EQ(octree.nLeaves, groups.nGroups) << name + ": wrong occupied leaf count";

        EXPECT_EQ(octree.nParticles, nParticles) << name + ": wrong particle count";

        const HostOctree host =
            copyOctree(octree);

        validateTopology(
            host,
            octree.nNodes);

        if (::testing::Test::HasFatalFailure())
        {
            cleanup();
            return;
        }

        /*
         * Copy group metadata so we can independently
         * verify occupied level-10 cells.
         */
        std::vector<std::uint32_t> uniqueKeys(
            static_cast<std::size_t>(
                groups.nGroups));

        std::vector<int> firstParticle(
            static_cast<std::size_t>(
                groups.nGroups));

        std::vector<int> particleCount(
            static_cast<std::size_t>(
                groups.nGroups));

        checkCuda(
            cudaMemcpy(
                uniqueKeys.data(),
                groups.uniqueKeys.data(),
                uniqueKeys.size() *
                    sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost),
            "copy unique keys");

        checkCuda(
            cudaMemcpy(
                firstParticle.data(),
                groups.firstParticle.data(),
                firstParticle.size() *
                    sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy group firstParticle");

        checkCuda(
            cudaMemcpy(
                particleCount.data(),
                groups.particleCount.data(),
                particleCount.size() *
                    sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy group particleCount");

        int occupiedLeaves = 0;

        for (int node = 0;
             node < octree.nNodes;
             ++node)
        {
            if (host.particleCount[node] <= 0)
            {
                EXPECT_EQ(host.firstParticle[node], -1)
                    << name + ": non-leaf contains particle metadata";

                continue;
            }

            ++occupiedLeaves;

            EXPECT_GT(host.level[node], 0)
                << name + ": occupied leaf cannot be the explicit root";

            EXPECT_LE(host.level[node], kOctreeMaxLevel)
                << name + ": occupied leaf exceeds maximum octree depth";
            
            bool matched = false;

            for (int group = 0;
                 group < groups.nGroups;
                 ++group)
            {
                if (host.prefix[node] !=
                    prefixAtLevel(
                        uniqueKeys[group],
                        host.level[node]))
                {
                    continue;
                }

                EXPECT_EQ(host.firstParticle[node], firstParticle[group])
                    << name + ": incorrect leaf offset";

                EXPECT_EQ(host.particleCount[node], particleCount[group])
                    << name + ": incorrect leaf count";

                matched = true;
                break;
            }

            EXPECT_TRUE(matched) << name +
                ": occupied leaf does not "
                "correspond to a Morton group";
        }

        EXPECT_EQ(occupiedLeaves, groups.nGroups)
            << name + ": not every Morton group has one occupied leaf";
    }
    catch (...)
    {
        cleanup();
        throw;
    }

    cleanup();
}

void testRandom()
{
    constexpr int n = 257;

    std::mt19937 generator(
        0xC0FFEEu);

    std::vector<std::uint32_t> keys(
        static_cast<std::size_t>(n));

    for (std::uint32_t& key : keys)
    {
        key =
            generator() &
            kMortonSpatialMask;
    }

    std::sort(
        keys.begin(),
        keys.end());

    // Deliberately introduce duplicates.
    for (int i = 7;
         i < n;
         i += 17)
    {
        keys[i] =
            keys[i - 1];
    }

    runCase(
        "random N=257",
        keys);
}

} // namespace

class SparseOctreeTest : public CudaTest
{
};

TEST_F(SparseOctreeTest, BuildsSingleOccupiedCell)
{
    runCase("single occupied cell", {0u});
}

TEST_F(SparseOctreeTest, BuildsDuplicateParticlesInOneCell)
{
    runCase(
        "duplicate particles in one cell",
        {7u, 7u, 7u, 7u});
}

TEST_F(SparseOctreeTest, BuildsTwoLevelOneOctants)
{
    runCase(
        "two level-one octants",
        {0u, 1u << 29});
}

TEST_F(SparseOctreeTest, BuildsSharedLevelOneCell)
{
    runCase(
        "shared level-one cell",
        {0u, 1u << 26});
}

TEST_F(SparseOctreeTest, BuildsMixedOccupiedCells)
{
    runCase(
        "mixed occupied cells",
        {
            0u,
            1u << 23,
            1u << 26,
            1u << 29,
            (1u << 29) | (1u << 26)
        });
}

TEST_F(SparseOctreeTest, BuildsDeterministicGeneratedInput)
{
    testRandom();
}
