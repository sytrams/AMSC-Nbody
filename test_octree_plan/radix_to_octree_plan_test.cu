#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

void require(
    bool condition,
    const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename T>
void requireVectorEqual(
    const std::vector<T>& actual,
    const std::vector<T>& expected,
    const std::string& label)
{
    require(
        actual.size() == expected.size(),
        label + ": vector sizes differ");

    for (std::size_t i = 0;
         i < actual.size();
         ++i)
    {
        if (actual[i] != expected[i])
        {
            throw std::runtime_error(
                label +
                ": mismatch at index " +
                std::to_string(i) +
                ", expected " +
                std::to_string(expected[i]) +
                ", got " +
                std::to_string(actual[i]));
        }
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
    require(
        !sortedKeys.empty(),
        name + ": empty key array");

    require(
        std::is_sorted(
            sortedKeys.begin(),
            sortedKeys.end()),
        name + ": keys are not sorted");

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

        require(
            plan.nRadixNodes ==
                nRadixNodes,
            name +
            ": incorrect radix-node count");

        require(
            plan.nOctreeNodes ==
                expectedOctreeNodes,
            name +
            ": incorrect octree-node count");

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

        requireVectorEqual(
            actualLevels,
            expectedLevels,
            name + " radix levels");

        requireVectorEqual(
            actualCounts,
            expectedCounts,
            name + " edge counts");

        requireVectorEqual(
            actualOffsets,
            expectedOffsets,
            name + " edge offsets");

        require(
            actualOffsets[0] == 0,
            name +
            ": exclusive scan must start at zero");

        for (int i = 1;
             i < nRadixNodes;
             ++i)
        {
            require(
                actualOffsets[i] ==
                    actualOffsets[i - 1] +
                    actualCounts[i - 1],
                name +
                ": offsets are not an exclusive scan");
        }

        int generatedNodes = 0;

        for (int count : actualCounts)
        {
            require(
                count >= 0,
                name +
                ": negative edge-node count");

            generatedNodes += count;
        }

        require(
            plan.nOctreeNodes ==
                generatedNodes + 1,
            name +
            ": total octree count is inconsistent");
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

    std::cout
        << "[PASS] "
        << name
        << '\n';
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

        bool exceptionThrown = false;

        try
        {
            buildRadixToOctreePlan(
                plan,
                tree,
                groups);
        }
        catch (const std::runtime_error&)
        {
            exceptionThrown = true;
        }

        require(
            exceptionThrown,
            "The conversion accepted a key "
            "outside the 30-bit Morton layout");
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

    std::cout
        << "[PASS] reject non-30-bit key\n";
}

} // namespace

int runRadixToOctreePlanTestSuite()
{
    try
    {
        int deviceCount = 0;

        checkCuda(
            cudaGetDeviceCount(
                &deviceCount),
            "cudaGetDeviceCount");

        require(
            deviceCount > 0,
            "No CUDA-capable GPU is available");

        runExactCase(
            "single occupied Morton cell",
            {0u},
            {10},
            {10},
            {0},
            11);

        runExactCase(
            "two different level-one octants",
            {
                0u,
                1u << 29
            },
            {
                10,
                10,
                0
            },
            {
                10,
                10,
                0
            },
            {
                0,
                10,
                20
            },
            21);

        runExactCase(
            "two cells sharing level one",
            {
                0u,
                1u << 26
            },
            {
                10,
                10,
                1
            },
            {
                9,
                9,
                1
            },
            {
                0,
                9,
                18
            },
            20);

        runExactCase(
            "duplicate particles in one Morton cell",
            {
                7u,
                7u,
                7u,
                7u
            },
            {
                10
            },
            {
                10
            },
            {
                0
            },
            11);

        testInvalidThirtyBitLayout();

        std::cout
            << "\nAll radix-to-octree planning "
            << "tests passed.\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "\n[FAIL] "
            << error.what()
            << '\n';

        return 1;
    }
}

TEST(RadixToOctreePlanCudaTest, CompleteSuite)
{
    EXPECT_EQ(runRadixToOctreePlanTestSuite(), 0);
}
