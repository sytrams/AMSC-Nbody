#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_test.hpp"
#include "morton_leaf_groups.hpp"
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

void expectRelativeNear(
    double actual,
    double expected,
    const std::string& label)
{
    const double scale =
        std::max({1.0, std::fabs(actual), std::fabs(expected)});

    EXPECT_NEAR(actual, expected, 2.0e-5 * scale)
        << label;
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

struct Aggregate
{
    double mass = 0.0f;
    double x = 0.0f;
    double y = 0.0f;
    double z = 0.0f;
};

Aggregate aggregateRange(
    int first,
    int count,
    const std::vector<std::uint32_t>& sortedIndices,
    const std::vector<double>& mass,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z)
{
    Aggregate result;

    for (int local = 0; local < count; ++local)
    {
        const int sortedPosition =
            first + local;

        const std::uint32_t particle =
            sortedIndices[sortedPosition];

        const double particleMass =
            mass[particle];

        result.mass += particleMass;
        result.x += particleMass * x[particle];
        result.y += particleMass * y[particle];
        result.z += particleMass * z[particle];
    }

    if (result.mass > 0.0f)
    {
        const double inverseMass =
            1.0f / result.mass;

        result.x *= inverseMass;
        result.y *= inverseMass;
        result.z *= inverseMass;
    }
    else
    {
        result.x = 0.0f;
        result.y = 0.0f;
        result.z = 0.0f;
    }

    return result;
}

void buildExpectedGroups(
    const std::vector<std::uint32_t>& sortedKeys,
    std::vector<std::uint32_t>& uniqueKeys,
    std::vector<int>& firstParticle,
    std::vector<int>& particleCount)
{
    for (int i = 0;
         i < static_cast<int>(sortedKeys.size());
         ++i)
    {
        if (i == 0 ||
            sortedKeys[i] != sortedKeys[i - 1])
        {
            uniqueKeys.push_back(sortedKeys[i]);
            firstParticle.push_back(i);
            particleCount.push_back(1);
        }
        else
        {
            ++particleCount.back();
        }
    }
}

void runCase(
    const std::string& name,
    const std::vector<std::uint32_t>& sortedKeys,
    const std::vector<std::uint32_t>& sortedIndices,
    const std::vector<double>& mass,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z)
{
    ASSERT_FALSE(sortedKeys.empty()) << name + ": empty Morton-key input";

    ASSERT_TRUE(std::is_sorted(
        sortedKeys.begin(),
        sortedKeys.end()))
        << name + ": Morton keys are not sorted";

    const int nParticles =
        static_cast<int>(sortedKeys.size());

    ASSERT_EQ(static_cast<int>(sortedIndices.size()), nParticles)
        << name + ": wrong sortedIndices size";

    ASSERT_EQ(static_cast<int>(mass.size()), nParticles)
        << name + ": wrong mass size";
    ASSERT_EQ(static_cast<int>(x.size()), nParticles)
        << name + ": wrong X size";
    ASSERT_EQ(static_cast<int>(y.size()), nParticles)
        << name + ": wrong Y size";
    ASSERT_EQ(static_cast<int>(z.size()), nParticles)
        << name + ": wrong Z size";

    std::vector<std::uint32_t> permutation =
        sortedIndices;

    std::sort(
        permutation.begin(),
        permutation.end());

    for (int i = 0; i < nParticles; ++i)
    {
        ASSERT_EQ(permutation[i], static_cast<std::uint32_t>(i))
            << name + ": sortedIndices is not a permutation";
    }

    std::vector<std::uint32_t> expectedUniqueKeys;
    std::vector<int> expectedFirstParticle;
    std::vector<int> expectedParticleCount;

    buildExpectedGroups(
        sortedKeys,
        expectedUniqueKeys,
        expectedFirstParticle,
        expectedParticleCount);

    DeviceArray<std::uint32_t> d_keys(sortedKeys);
    DeviceArray<std::uint32_t> d_indices(sortedIndices);
    DeviceArray<double> d_mass(mass);
    DeviceArray<double> d_x(x);
    DeviceArray<double> d_y(y);
    DeviceArray<double> d_z(z);

    MortonLeafGroups groups{};
    Tree tree{};

    try
    {
        allocateMortonLeafGroups(
            groups,
            nParticles);

        buildMortonLeafGroups(
            groups,
            d_keys.get(),
            nParticles);

        EXPECT_EQ(
            groups.nGroups,
            static_cast<int>(expectedUniqueKeys.size()))
            << name + ": wrong number of Morton groups";

        allocateTree(
            tree,
            groups.nGroups);

        buildTreeFromMortonGroups(
            tree,
            groups);

        computeGroupedCenterOfMass(
            tree,
            groups,
            d_indices.get(),
            d_mass.get(),
            d_x.get(),
            d_y.get(),
            d_z.get());

        EXPECT_EQ(tree.nLeaves, groups.nGroups)
            << name + ": tree leaf count differs from group count";

        const int nLeaves = tree.nLeaves;
        const int totalNodes =
            2 * nLeaves - 1;

        std::vector<std::uint32_t> actualUniqueKeys(
            static_cast<std::size_t>(nLeaves));

        std::vector<int> actualFirstParticle(
            static_cast<std::size_t>(nLeaves));

        std::vector<int> actualParticleCount(
            static_cast<std::size_t>(nLeaves));

        checkCuda(
            cudaMemcpy(
                actualUniqueKeys.data(),
                groups.uniqueKeys,
                static_cast<std::size_t>(nLeaves) *
                    sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost),
            "copy unique keys");

        checkCuda(
            cudaMemcpy(
                actualFirstParticle.data(),
                groups.firstParticle,
                static_cast<std::size_t>(nLeaves) *
                    sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy firstParticle");

        checkCuda(
            cudaMemcpy(
                actualParticleCount.data(),
                groups.particleCount,
                static_cast<std::size_t>(nLeaves) *
                    sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy particleCount");

        EXPECT_EQ(actualUniqueKeys, expectedUniqueKeys)
            << name + ": unique Morton keys differ";

        EXPECT_EQ(actualFirstParticle, expectedFirstParticle)
            << name + ": group offsets differ";

        EXPECT_EQ(actualParticleCount, expectedParticleCount)
            << name + ": group counts differ";

        std::vector<double> treeMass(
            static_cast<std::size_t>(totalNodes));

        std::vector<double> treeComX(
            static_cast<std::size_t>(totalNodes));

        std::vector<double> treeComY(
            static_cast<std::size_t>(totalNodes));

        std::vector<double> treeComZ(
            static_cast<std::size_t>(totalNodes));

        checkCuda(
            cudaMemcpy(
                treeMass.data(),
                tree.mass,
                static_cast<std::size_t>(totalNodes) *
                    sizeof(double),
                cudaMemcpyDeviceToHost),
            "copy tree mass");

        checkCuda(
            cudaMemcpy(
                treeComX.data(),
                tree.comX,
                static_cast<std::size_t>(totalNodes) *
                    sizeof(double),
                cudaMemcpyDeviceToHost),
            "copy tree comX");

        checkCuda(
            cudaMemcpy(
                treeComY.data(),
                tree.comY,
                static_cast<std::size_t>(totalNodes) *
                    sizeof(double),
                cudaMemcpyDeviceToHost),
            "copy tree comY");

        checkCuda(
            cudaMemcpy(
                treeComZ.data(),
                tree.comZ,
                static_cast<std::size_t>(totalNodes) *
                    sizeof(double),
                cudaMemcpyDeviceToHost),
            "copy tree comZ");

        for (int leaf = 0;
             leaf < nLeaves;
             ++leaf)
        {
            const Aggregate expected =
                aggregateRange(
                    expectedFirstParticle[leaf],
                    expectedParticleCount[leaf],
                    sortedIndices,
                    mass,
                    x,
                    y,
                    z);

            expectRelativeNear(treeMass[leaf], expected.mass, name + ": leaf mass");

            expectRelativeNear(treeComX[leaf], expected.x, name + ": leaf comX");

            expectRelativeNear(treeComY[leaf], expected.y, name + ": leaf comY");

            expectRelativeNear(treeComZ[leaf], expected.z, name + ": leaf comZ");
        }

        const Aggregate expectedRoot =
            aggregateRange(
                0,
                nParticles,
                sortedIndices,
                mass,
                x,
                y,
                z);

        const int root =
            (nLeaves == 1)
                ? 0
                : nLeaves;

        expectRelativeNear(treeMass[root], expectedRoot.mass, name + ": root mass");

        expectRelativeNear(treeComX[root], expectedRoot.x, name + ": root comX");

        expectRelativeNear(treeComY[root], expectedRoot.y, name + ": root comY");

        expectRelativeNear(treeComZ[root], expectedRoot.z, name + ": root comZ");

        std::vector<int> parent(
            static_cast<std::size_t>(totalNodes));

        checkCuda(
            cudaMemcpy(
                parent.data(),
                tree.parent,
                static_cast<std::size_t>(totalNodes) *
                    sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy tree parent");

        EXPECT_EQ(parent[root], -1) << name + ": root parent is not -1";

        if (nLeaves > 1)
        {
            std::vector<int> rangeFirst(
                static_cast<std::size_t>(
                    nLeaves - 1));

            std::vector<int> rangeLast(
                static_cast<std::size_t>(
                    nLeaves - 1));

            std::vector<int> prefixLength(
                static_cast<std::size_t>(
                    nLeaves - 1));

            std::vector<int> visitCount(
                static_cast<std::size_t>(
                    nLeaves - 1));

            checkCuda(
                cudaMemcpy(
                    rangeFirst.data(),
                    tree.rangeFirst,
                    static_cast<std::size_t>(
                        nLeaves - 1) *
                        sizeof(int),
                    cudaMemcpyDeviceToHost),
                "copy rangeFirst");

            checkCuda(
                cudaMemcpy(
                    rangeLast.data(),
                    tree.rangeLast,
                    static_cast<std::size_t>(
                        nLeaves - 1) *
                        sizeof(int),
                    cudaMemcpyDeviceToHost),
                "copy rangeLast");

            checkCuda(
                cudaMemcpy(
                    prefixLength.data(),
                    tree.prefixLength,
                    static_cast<std::size_t>(
                        nLeaves - 1) *
                        sizeof(int),
                    cudaMemcpyDeviceToHost),
                "copy prefixLength");

            checkCuda(
                cudaMemcpy(
                    visitCount.data(),
                    tree.visitCount,
                    static_cast<std::size_t>(
                        nLeaves - 1) *
                        sizeof(int),
                    cudaMemcpyDeviceToHost),
                "copy visitCount");

            // In the Karras indexing used by the current
            // implementation, internal local index 0 is root.
            EXPECT_EQ(rangeFirst[0], 0)
                << name + ": root range does not start at zero";

            EXPECT_EQ(rangeLast[0], nLeaves - 1)
                << name + ": root range does not cover all leaves";

            for (int prefix : prefixLength)
            {
                EXPECT_TRUE(prefix >= 0 &&
                    prefix <= 32) << name +
                    ": unique-key tree contains "
                    "a virtual index suffix";
            }

            for (int count : visitCount)
            {
                EXPECT_EQ(count, 2)
                    << name + ": an internal node was not "
                       "completed by exactly two children";
            }
        }
    }
    catch (...)
    {
        freeTree(tree);
        freeMortonLeafGroups(groups);
        throw;
    }

    freeTree(tree);
    freeMortonLeafGroups(groups);
}

void testGeneratedCase()
{
    constexpr int n = 257;

    std::vector<std::uint32_t> keys(
        static_cast<std::size_t>(n));

    std::vector<std::uint32_t> indices(
        static_cast<std::size_t>(n));

    std::vector<double> mass(
        static_cast<std::size_t>(n));

    std::vector<double> x(
        static_cast<std::size_t>(n));

    std::vector<double> y(
        static_cast<std::size_t>(n));

    std::vector<double> z(
        static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i)
    {
        // Mostly groups of three equal keys.
        keys[i] =
            static_cast<std::uint32_t>(i / 3);

        indices[i] =
            static_cast<std::uint32_t>(
                n - 1 - i);

        mass[i] =
            0.5f +
            static_cast<double>(i % 11) *
                0.25f;

        x[i] =
            static_cast<double>(i) * 0.1f;

        y[i] =
            static_cast<double>((i * i) % 31);

        z[i] =
            -static_cast<double>(i) * 0.05f;
    }

    runCase(
        "generated grouped N=257",
        keys,
        indices,
        mass,
        x,
        y,
        z);
}

} // namespace

class RadixTreeGroupsTest : public CudaTest
{
};

TEST_F(RadixTreeGroupsTest, HandlesSingleParticle)
{
    runCase(
        "single particle",
        {42u},
        {0u},
        {2.0f},
        {1.0f},
        {2.0f},
        {3.0f});
}

TEST_F(RadixTreeGroupsTest, HandlesAllDuplicateMortonKeys)
{
    runCase(
        "all duplicate Morton keys",
        {7u, 7u, 7u, 7u},
        {3u, 0u, 2u, 1u},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {0.0f, 10.0f, 20.0f, 30.0f},
        {5.0f, 4.0f, 3.0f, 2.0f},
        {-1.0f, 0.0f, 1.0f, 2.0f});
}

TEST_F(RadixTreeGroupsTest, HandlesMixedMortonGroups)
{
    runCase(
        "mixed Morton groups",
        {1u, 1u, 2u, 4u, 4u, 4u, 9u},
        {6u, 2u, 4u, 0u, 5u, 1u, 3u},
        {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f},
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f},
        {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
}

TEST_F(RadixTreeGroupsTest, HandlesAllUniqueMortonKeys)
{
    runCase(
        "all unique Morton keys",
        {0u, 4u, 8u, 12u},
        {2u, 0u, 3u, 1u},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {-1.0f, -2.0f, -3.0f, -4.0f},
        {5.0f, 15.0f, 25.0f, 35.0f});
}

TEST_F(RadixTreeGroupsTest, HandlesGeneratedGroupedInput)
{
    testGeneratedCase();
}
