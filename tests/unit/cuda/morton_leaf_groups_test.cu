#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "cuda_test.hpp"
#include "morton_leaf_groups.hpp"

namespace 
{
    static_assert(
        !std::is_copy_constructible_v<
            MortonLeafGroups>);

    static_assert(
        !std::is_copy_assignable_v<
            MortonLeafGroups>);

    static_assert(
        std::is_move_constructible_v<
            MortonLeafGroups>);

    static_assert(
        std::is_move_assignable_v<
            MortonLeafGroups>);

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

    void validateInvariants(
        const std::vector<std::uint32_t>& sortedKeys,
        const std::vector<std::uint32_t>& uniqueKeys,
        const std::vector<int>& firstParticle,
        const std::vector<int>& particleCount)
    {
        const int nGroups =
            static_cast<int>(uniqueKeys.size());

        ASSERT_EQ(static_cast<int>(firstParticle.size()), nGroups)
            << "firstParticle size does not match nGroups";

        ASSERT_EQ(static_cast<int>(particleCount.size()), nGroups)
            << "particleCount size does not match nGroups";

        ASSERT_GT(nGroups, 0)
            << "There must be at least one Morton group";

        EXPECT_EQ(firstParticle[0], 0)
            << "The first Morton group must start at offset zero";

        const int totalParticleCount =
            std::accumulate(
                particleCount.begin(),
                particleCount.end(),
                0);

        EXPECT_EQ(totalParticleCount, static_cast<int>(sortedKeys.size()))
            << "The sum of group counts does not match particle count";

        for (int group = 0; group < nGroups; ++group)
        {
            ASSERT_GT(particleCount[group], 0)
                << "A Morton group has non-positive size";

            if (group + 1 < nGroups)
            {
                EXPECT_LT(uniqueKeys[group], uniqueKeys[group + 1])
                    << "Unique Morton keys are not strictly increasing";

                EXPECT_EQ(
                    firstParticle[group + 1],
                    firstParticle[group] + particleCount[group])
                    << "Morton group offsets are not contiguous";
            }

            const int first = firstParticle[group];
            const int count = particleCount[group];

            ASSERT_GE(first, 0)
                << "A Morton group has a negative starting offset";

            ASSERT_LE(first + count, static_cast<int>(sortedKeys.size()))
                << "A Morton group exceeds the input range";

            for (int localParticle = 0;
                localParticle < count;
                ++localParticle)
            {
                EXPECT_EQ(
                    sortedKeys[first + localParticle],
                    uniqueKeys[group])
                    << "A group contains a key different "
                       "from its unique Morton key";
            }
        }
    }

    void runCase(
        const std::string& name,
        const std::vector<std::uint32_t>& sortedKeys,
        const std::vector<std::uint32_t>& expectedUniqueKeys,
        const std::vector<int>& expectedFirstParticle,
        const std::vector<int>& expectedParticleCount)
    {
        std::uint32_t* d_sortedKeys = nullptr;
        MortonLeafGroups groups{};

        try
        {
            const std::size_t inputBytes =
                sortedKeys.size() * sizeof(std::uint32_t);
            
            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&d_sortedKeys),
                    inputBytes),
                "cudaMalloc test sorted keys");

            checkCuda(
                cudaMemcpy(
                    d_sortedKeys,
                    sortedKeys.data(),
                    inputBytes,
                    cudaMemcpyHostToDevice),
                "copy test sorted keys to device");

            allocateMortonLeafGroups(
                groups,
                static_cast<int>(sortedKeys.size()));

            buildMortonLeafGroups(
                groups,
                d_sortedKeys,
                static_cast<int>(sortedKeys.size()));

            EXPECT_EQ(groups.nParticles, static_cast<int>(sortedKeys.size()))
                << "groups.nParticles is incorrect";

            EXPECT_EQ(groups.nGroups, static_cast<int>(expectedUniqueKeys.size()))
                << "groups.nGroups is incorrect";

            EXPECT_EQ(groups.capacity, static_cast<int>(sortedKeys.size()))
                << "groups.capacity is incorrect";

            std::vector<std::uint32_t> actualUniqueKeys(
                static_cast<std::size_t>(groups.nGroups));

            std::vector<int> actualFirstParticle(
                static_cast<std::size_t>(groups.nGroups));

            std::vector<int> actualParticleCount(
                static_cast<std::size_t>(groups.nGroups));

            const std::size_t keyBytes =
                static_cast<std::size_t>(groups.nGroups) *
                sizeof(std::uint32_t);

            const std::size_t integerBytes =
                static_cast<std::size_t>(groups.nGroups) *
                sizeof(int);

            checkCuda(
                cudaMemcpy(
                    actualUniqueKeys.data(),
                    groups.uniqueKeys.data(),
                    keyBytes,
                    cudaMemcpyDeviceToHost),
                "copy unique keys to host");

            checkCuda(
                cudaMemcpy(
                    actualFirstParticle.data(),
                    groups.firstParticle.data(),
                    integerBytes,
                    cudaMemcpyDeviceToHost),
                "copy group offsets to host");

            checkCuda(
                cudaMemcpy(
                    actualParticleCount.data(),
                    groups.particleCount.data(),
                    integerBytes,
                    cudaMemcpyDeviceToHost),
                "copy group counts to host");

            EXPECT_EQ(actualUniqueKeys, expectedUniqueKeys) << name + " uniqueKeys";

            EXPECT_EQ(actualFirstParticle, expectedFirstParticle) << name + " firstParticle";

            EXPECT_EQ(actualParticleCount, expectedParticleCount) << name + " particleCount";

            validateInvariants(
                sortedKeys,
                actualUniqueKeys,
                actualFirstParticle,
                actualParticleCount);
        }
        catch (...)
        {
            freeMortonLeafGroups(groups);
            cudaFree(d_sortedKeys);
            throw;
        }

        freeMortonLeafGroups(groups);
        cudaFree(d_sortedKeys);
    }

    void testCapacityValidation()
    {
        std::uint32_t* d_sortedKeys = nullptr;
        MortonLeafGroups groups{};

        try
        {
            const std::vector<std::uint32_t> sortedKeys{
                1u,
                2u,
                3u
            };

            checkCuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&d_sortedKeys),
                    sortedKeys.size() * sizeof(std::uint32_t)),
                "cudaMalloc capacity-test keys");

            checkCuda(
                cudaMemcpy(
                    d_sortedKeys,
                    sortedKeys.data(),
                    sortedKeys.size() * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice),
                "copy capacity-test keys");

            // Gli array possono contenere soltanto due elementi.
            allocateMortonLeafGroups(groups, 2);

            EXPECT_THROW(
                buildMortonLeafGroups(
                    groups,
                    d_sortedKeys,
                    3),
                std::invalid_argument);
        }
        catch (...)
        {
            freeMortonLeafGroups(groups);
            cudaFree(d_sortedKeys);
            throw;
        }

        freeMortonLeafGroups(groups);
        cudaFree(d_sortedKeys);
    }
} // namespace

class MortonLeafGroupsTest : public CudaTest
{
};

TEST_F(MortonLeafGroupsTest, GroupsSingleKey)
{
    runCase("single key", {42u}, {42u}, {0}, {1});
}

TEST_F(MortonLeafGroupsTest, PreservesAllUniqueKeys)
{
    runCase(
        "all unique keys",
        {2u, 5u, 8u, 10u},
        {2u, 5u, 8u, 10u},
        {0, 1, 2, 3},
        {1, 1, 1, 1});
}

TEST_F(MortonLeafGroupsTest, CombinesDuplicateKeys)
{
    runCase(
        "all duplicate keys",
        {7u, 7u, 7u, 7u},
        {7u},
        {0},
        {4});
}

TEST_F(MortonLeafGroupsTest, GroupsMixedKeys)
{
    runCase(
        "mixed key groups",
        {1u, 1u, 2u, 4u, 4u, 4u, 9u},
        {1u, 2u, 4u, 9u},
        {0, 2, 3, 6},
        {2, 1, 3, 1});
}

TEST_F(MortonLeafGroupsTest, HandlesMinimumAndMaximumKeys)
{
    runCase(
        "minimum and maximum keys",
        {0u, 0u, 1u, 0xFFFFFFFFu, 0xFFFFFFFFu},
        {0u, 1u, 0xFFFFFFFFu},
        {0, 2, 3},
        {2, 1, 2});
}

TEST_F(MortonLeafGroupsTest, RejectsInputLargerThanCapacity)
{
    testCapacityValidation();
}
