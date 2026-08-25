#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_test.hpp"
#include "morton_leaf_groups.hpp"
#include "radix_to_octree.hpp"
#include "tree_builder.hpp"

namespace {

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
                "copy DeviceArray to device");
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

    DeviceArray(
        const DeviceArray&) = delete;

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

void runExactCase(
    const std::string& name,
    const std::vector<std::uint32_t>& sortedKeys,
    const std::vector<int>& expectedLevels,
    const std::vector<int>& expectedCounts,
    const std::vector<int>& expectedOffsets,
    int expectedOctreeNodes)
{
    ASSERT_FALSE(sortedKeys.empty()) << name + ": empty key array";

    ASSERT_TRUE(std::is_sorted(
        sortedKeys.begin(),
        sortedKeys.end()))
        << name + ": keys are not sorted";

    DeviceArray<std::uint32_t>
        deviceKeys(sortedKeys);

    MortonLeafGroups groups{};
    Tree tree{};
    RadixToOctreePlan plan{};

    try
    {
        allocateMortonLeafGroups(
            groups,
            static_cast<int>(
                sortedKeys.size()));

        buildMortonLeafGroups(
            groups,
            deviceKeys.get(),
            static_cast<int>(
                sortedKeys.size()));

        allocateTree(
            tree,
            groups.nGroups);

        buildTreeFromMortonGroups(
            tree,
            groups);

        const int nRadixNodes =
            2 * groups.nGroups - 1;

        allocateRadixToOctreePlan(
            plan,
            nRadixNodes);

        buildRadixToOctreePlan(
            plan,
            tree,
            groups);

        EXPECT_EQ(plan.nRadixNodes, nRadixNodes)
            << name + ": incorrect radix-node count";

        EXPECT_EQ(plan.nOctreeNodes, expectedOctreeNodes)
            << name + ": incorrect octree-node count";

        std::vector<int> actualLevels(
            static_cast<std::size_t>(
                nRadixNodes));

        std::vector<int> actualCounts(
            static_cast<std::size_t>(
                nRadixNodes));

        std::vector<int> actualOffsets(
            static_cast<std::size_t>(
                nRadixNodes));

        const std::size_t bytes =
            static_cast<std::size_t>(
                nRadixNodes) *
            sizeof(int);

        checkCuda(
            cudaMemcpy(
                actualLevels.data(),
                plan.radixLevel,
                bytes,
                cudaMemcpyDeviceToHost),
            "copy radix levels");

        checkCuda(
            cudaMemcpy(
                actualCounts.data(),
                plan.edgeNodeCount,
                bytes,
                cudaMemcpyDeviceToHost),
            "copy edge counts");

        checkCuda(
            cudaMemcpy(
                actualOffsets.data(),
                plan.edgeNodeOffset,
                bytes,
                cudaMemcpyDeviceToHost),
            "copy edge offsets");

        EXPECT_EQ(actualLevels, expectedLevels) << name + " radix levels";

        EXPECT_EQ(actualCounts, expectedCounts) << name + " edge counts";

        EXPECT_EQ(actualOffsets, expectedOffsets) << name + " edge offsets";

        EXPECT_EQ(actualOffsets[0], 0)
            << name + ": exclusive scan must start at zero";

        for (int i = 1;
             i < nRadixNodes;
             ++i)
        {
            EXPECT_EQ(
                actualOffsets[i],
                actualOffsets[i - 1] + actualCounts[i - 1])
                << name + ": offsets are not an exclusive scan";
        }

        int generatedNodes = 0;

        for (int count : actualCounts)
        {
            EXPECT_GE(count, 0)
                << name + ": negative edge-node count";

            generatedNodes += count;
        }

        EXPECT_EQ(plan.nOctreeNodes, generatedNodes + 1)
            << name + ": total octree count is inconsistent";
    }
    catch (...)
    {
        freeRadixToOctreePlan(plan);
        freeTree(tree);
        freeMortonLeafGroups(groups);
        throw;
    }

    freeRadixToOctreePlan(plan);
    freeTree(tree);
    freeMortonLeafGroups(groups);
}

void testInvalidThirtyBitLayout()
{
    const std::vector<std::uint32_t> keys{
        0u,
        0x80000000u
    };

    DeviceArray<std::uint32_t>
        deviceKeys(keys);

    MortonLeafGroups groups{};
    Tree tree{};
    RadixToOctreePlan plan{};

    try
    {
        allocateMortonLeafGroups(
            groups,
            2);

        buildMortonLeafGroups(
            groups,
            deviceKeys.get(),
            2);

        allocateTree(
            tree,
            groups.nGroups);

        buildTreeFromMortonGroups(
            tree,
            groups);

        allocateRadixToOctreePlan(
            plan,
            2 * groups.nGroups - 1);

        EXPECT_THROW(
            buildRadixToOctreePlan(
                plan,
                tree,
                groups),
            std::runtime_error);
    }
    catch (...)
    {
        freeRadixToOctreePlan(plan);
        freeTree(tree);
        freeMortonLeafGroups(groups);
        throw;
    }

    freeRadixToOctreePlan(plan);
    freeTree(tree);
    freeMortonLeafGroups(groups);
}

} // namespace

class RadixToOctreeTest : public CudaTest
{
};

TEST_F(RadixToOctreeTest, PlansSingleOccupiedMortonCell)
{
    runExactCase(
        "single occupied Morton cell",
        {0u},
        {10},
        {10},
        {0},
        11);
}

TEST_F(RadixToOctreeTest, PlansTwoLevelOneOctants)
{
    runExactCase(
        "two different level-one octants",
        {0u, 1u << 29},
        {10, 10, 0},
        {10, 10, 0},
        {0, 10, 20},
        21);
}

TEST_F(RadixToOctreeTest, PlansCellsSharingLevelOne)
{
    runExactCase(
        "two cells sharing level one",
        {0u, 1u << 26},
        {10, 10, 1},
        {9, 9, 1},
        {0, 9, 18},
        20);
}

TEST_F(RadixToOctreeTest, PlansDuplicateParticlesInOneCell)
{
    runExactCase(
        "duplicate particles in one Morton cell",
        {7u, 7u, 7u, 7u},
        {10},
        {10},
        {0},
        11);
}

TEST_F(RadixToOctreeTest, RejectsKeysOutsideThirtyBitLayout)
{
    testInvalidThirtyBitLayout();
}
