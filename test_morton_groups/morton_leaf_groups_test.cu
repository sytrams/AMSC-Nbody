#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "morton_leaf_groups.hpp"

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
    const std::string& description)
{
    if (actual.size() != expected.size())
    {
        throw std::runtime_error(
            description + ": vector sizes differ");
    }

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (actual[i] != expected[i])
        {
            throw std::runtime_error(
                description +
                ": mismatch at index " +
                std::to_string(i));
        }
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

    require(
        static_cast<int>(firstParticle.size()) == nGroups,
        "firstParticle size does not match nGroups");

    require(
        static_cast<int>(particleCount.size()) == nGroups,
        "particleCount size does not match nGroups");

    require(
        nGroups > 0,
        "There must be at least one Morton group");

    require(
        firstParticle[0] == 0,
        "The first Morton group must start at offset zero");

    const int totalParticleCount =
        std::accumulate(
            particleCount.begin(),
            particleCount.end(),
            0);

    require(
        totalParticleCount ==
            static_cast<int>(sortedKeys.size()),
        "The sum of group counts does not match particle count");

    for (int group = 0; group < nGroups; ++group)
    {
        require(
            particleCount[group] > 0,
            "A Morton group has non-positive size");

        if (group + 1 < nGroups)
        {
            require(
                uniqueKeys[group] < uniqueKeys[group + 1],
                "Unique Morton keys are not strictly increasing");

            require(
                firstParticle[group + 1] ==
                    firstParticle[group] +
                    particleCount[group],
                "Morton group offsets are not contiguous");
        }

        const int first = firstParticle[group];
        const int count = particleCount[group];

        require(
            first >= 0,
            "A Morton group has a negative starting offset");

        require(
            first + count <=
                static_cast<int>(sortedKeys.size()),
            "A Morton group exceeds the input range");

        for (int localParticle = 0;
             localParticle < count;
             ++localParticle)
        {
            require(
                sortedKeys[first + localParticle] ==
                    uniqueKeys[group],
                "A group contains a key different "
                "from its unique Morton key");
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

        require(
            groups.nParticles ==
                static_cast<int>(sortedKeys.size()),
            "groups.nParticles is incorrect");

        require(
            groups.nGroups ==
                static_cast<int>(expectedUniqueKeys.size()),
            "groups.nGroups is incorrect");

        require(
            groups.capacity ==
                static_cast<int>(sortedKeys.size()),
            "groups.capacity is incorrect");

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
                groups.uniqueKeys,
                keyBytes,
                cudaMemcpyDeviceToHost),
            "copy unique keys to host");

        checkCuda(
            cudaMemcpy(
                actualFirstParticle.data(),
                groups.firstParticle,
                integerBytes,
                cudaMemcpyDeviceToHost),
            "copy group offsets to host");

        checkCuda(
            cudaMemcpy(
                actualParticleCount.data(),
                groups.particleCount,
                integerBytes,
                cudaMemcpyDeviceToHost),
            "copy group counts to host");

        requireVectorEqual(
            actualUniqueKeys,
            expectedUniqueKeys,
            name + " uniqueKeys");

        requireVectorEqual(
            actualFirstParticle,
            expectedFirstParticle,
            name + " firstParticle");

        requireVectorEqual(
            actualParticleCount,
            expectedParticleCount,
            name + " particleCount");

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

    std::cout
        << "[PASS] "
        << name
        << '\n';
}

} // namespace

int main()
{
    try
    {
        runCase(
            "single key",
            {42u},
            {42u},
            {0},
            {1});

        runCase(
            "all unique keys",
            {2u, 5u, 8u, 10u},
            {2u, 5u, 8u, 10u},
            {0, 1, 2, 3},
            {1, 1, 1, 1});

        runCase(
            "all duplicate keys",
            {7u, 7u, 7u, 7u},
            {7u},
            {0},
            {4});

        runCase(
            "mixed key groups",
            {1u, 1u, 2u, 4u, 4u, 4u, 9u},
            {1u, 2u, 4u, 9u},
            {0, 2, 3, 6},
            {2, 1, 3, 1});

        runCase(
            "minimum and maximum keys",
            {
                0u,
                0u,
                1u,
                0xFFFFFFFFu,
                0xFFFFFFFFu
            },
            {
                0u,
                1u,
                0xFFFFFFFFu
            },
            {0, 2, 3},
            {2, 1, 2});

        std::cout
            << "\nAll Morton leaf-group tests passed.\n";

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "[FAIL] "
            << error.what()
            << '\n';

        return 1;
    }
}