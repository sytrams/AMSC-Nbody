#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>

#include "cuda_test.hpp"
#include "morton_leaf_groups.hpp"
#include "octree_builder.hpp"
#include "radix_to_octree.hpp"
#include "tree_builder.hpp"
#include "octree_physics.hpp"

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
    constexpr double relativeTolerance = 1.0e-5;
    const double scale =
        std::max({1.0, std::abs(actual), std::abs(expected)});

    EXPECT_NEAR(actual, expected, relativeTolerance * scale)
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

    std::vector<double> mass;
    std::vector<double> comX;
    std::vector<double> comY;
    std::vector<double> comZ;

    std::vector<int> pendingChildren;
    std::vector<int> childCount;
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

    host.mass.resize(n);
    host.comX.resize(n);
    host.comY.resize(n);
    host.comZ.resize(n);

    host.pendingChildren.resize(n);
    host.childCount.resize(n);

    checkCuda(
        cudaMemcpy(
            host.children.data(),
            octree.children,
            n * 8 * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree children");

    checkCuda(
        cudaMemcpy(
            host.parent.data(),
            octree.parent,
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree parent");

    checkCuda(
        cudaMemcpy(
            host.level.data(),
            octree.level,
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree levels");

    checkCuda(
        cudaMemcpy(
            host.prefix.data(),
            octree.prefix,
            n * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost),
        "copy octree prefixes");

    checkCuda(
        cudaMemcpy(
            host.firstParticle.data(),
            octree.firstParticle,
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy firstParticle");

    checkCuda(
        cudaMemcpy(
            host.particleCount.data(),
            octree.particleCount,
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy particleCount");

    checkCuda(
        cudaMemcpy(
            host.mass.data(),
            octree.mass,
            n * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy octree mass");

    checkCuda(
        cudaMemcpy(
            host.comX.data(),
            octree.comX,
            n * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy octree comX");

    checkCuda(
        cudaMemcpy(
            host.comY.data(),
            octree.comY,
            n * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy octree comY");

    checkCuda(
        cudaMemcpy(
            host.comZ.data(),
            octree.comZ,
            n * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy octree comZ");

    checkCuda(
        cudaMemcpy(
            host.pendingChildren.data(),
            octree.pendingChildren,
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree pendingChildren");

    checkCuda(
        cudaMemcpy(
            host.childCount.data(),
            octree.childCount,
            n * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy octree childCount");

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
    ASSERT_EQ(static_cast<int>(tree.mass.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.comX.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.comY.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.comZ.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.pendingChildren.size()), nNodes);
    ASSERT_EQ(static_cast<int>(tree.childCount.size()), nNodes);

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

void validatePhysics(
    const HostOctree& tree,
    int nNodes,
    int nParticles,
    const std::vector<std::uint32_t>& sortedIndices,
    const std::vector<double>& particleMass,
    const std::vector<double>& positionX,
    const std::vector<double>& positionY,
    const std::vector<double>& positionZ)
{
    ASSERT_GT(nNodes, 0);
    ASSERT_EQ(static_cast<int>(sortedIndices.size()), nParticles);
    ASSERT_EQ(static_cast<int>(particleMass.size()), nParticles);
    ASSERT_EQ(static_cast<int>(positionX.size()), nParticles);
    ASSERT_EQ(static_cast<int>(positionY.size()), nParticles);
    ASSERT_EQ(static_cast<int>(positionZ.size()), nParticles);

    /*
     * 1. Check every occupied leaf independently
     *    from the original particle data.
     */
    for (int node = 0;
         node < nNodes;
         ++node)
    {
        if (tree.particleCount[node] <= 0)
            continue;

        EXPECT_EQ(tree.childCount[node], 0) << "Occupied leaf has children";

        EXPECT_EQ(tree.pendingChildren[node], 0) << "Occupied leaf has pending children";

        const int first =
            tree.firstParticle[node];

        const int count =
            tree.particleCount[node];

        ASSERT_GE(first, 0) << "Invalid occupied-leaf particle range";
        ASSERT_GT(count, 0) << "Invalid occupied-leaf particle count";
        ASSERT_LE(first + count, nParticles)
            << "Invalid occupied-leaf particle range";

        double expectedMass = 0.0f;
        double weightedX = 0.0f;
        double weightedY = 0.0f;
        double weightedZ = 0.0f;

        for (int local = 0;
             local < count;
             ++local)
        {
            const int sortedPosition =
                first + local;

            const std::uint32_t particle =
                sortedIndices[sortedPosition];

            ASSERT_LT(
                particle,
                static_cast<std::uint32_t>(nParticles))
                << "Invalid sorted particle index";

            const double mass =
                particleMass[particle];

            expectedMass += mass;

            weightedX +=
                mass * positionX[particle];

            weightedY +=
                mass * positionY[particle];

            weightedZ +=
                mass * positionZ[particle];
        }

        expectRelativeNear(tree.mass[node], expectedMass, "Incorrect leaf mass");

        if (expectedMass > 0.0f)
        {
            expectRelativeNear(tree.comX[node], weightedX / expectedMass, "Incorrect leaf comX");

            expectRelativeNear(tree.comY[node], weightedY / expectedMass, "Incorrect leaf comY");

            expectRelativeNear(tree.comZ[node], weightedZ / expectedMass, "Incorrect leaf comZ");
        }
        else
        {
            expectRelativeNear(tree.comX[node], 0.0f, "Zero-mass leaf comX");

            expectRelativeNear(tree.comY[node], 0.0f, "Zero-mass leaf comY");

            expectRelativeNear(tree.comZ[node], 0.0f, "Zero-mass leaf comZ");
        }
    }

    /*
     * 2. Independently recompute every internal node
     *    from its children.
     */
    for (int node = 0;
         node < nNodes;
         ++node)
    {
        int actualChildCount = 0;

        double expectedMass = 0.0f;
        double weightedX = 0.0f;
        double weightedY = 0.0f;
        double weightedZ = 0.0f;

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

            ++actualChildCount;

            const double childMass =
                tree.mass[child];

            expectedMass +=
                childMass;

            weightedX +=
                childMass *
                tree.comX[child];

            weightedY +=
                childMass *
                tree.comY[child];

            weightedZ +=
                childMass *
                tree.comZ[child];
        }

        EXPECT_EQ(tree.childCount[node], actualChildCount)
            << "Incorrect childCount";

        if (actualChildCount == 0)
            continue;

        EXPECT_EQ(tree.pendingChildren[node], actualChildCount)
            << "Internal node did not receive all child completions";

        expectRelativeNear(tree.mass[node], expectedMass, "Incorrect internal-node mass");

        if (expectedMass > 0.0f)
        {
            expectRelativeNear(tree.comX[node], weightedX / expectedMass, "Incorrect internal comX");

            expectRelativeNear(tree.comY[node], weightedY / expectedMass, "Incorrect internal comY");

            expectRelativeNear(tree.comZ[node], weightedZ / expectedMass, "Incorrect internal comZ");
        }
        else
        {
            expectRelativeNear(tree.comX[node], 0.0f, "Zero-mass internal comX");

            expectRelativeNear(tree.comY[node], 0.0f, "Zero-mass internal comY");

            expectRelativeNear(tree.comZ[node], 0.0f, "Zero-mass internal comZ");
        }
    }

    /*
     * 3. Completely independent reference calculation
     *    for the root from ALL original particles.
     */
    double totalMass = 0.0f;
    double weightedX = 0.0f;
    double weightedY = 0.0f;
    double weightedZ = 0.0f;

    for (int particle = 0;
         particle < nParticles;
         ++particle)
    {
        const double mass =
            particleMass[particle];

        totalMass += mass;

        weightedX +=
            mass * positionX[particle];

        weightedY +=
            mass * positionY[particle];

        weightedZ +=
            mass * positionZ[particle];
    }

    expectRelativeNear(tree.mass[0], totalMass, "Incorrect root mass");

    if (totalMass > 0.0f)
    {
        expectRelativeNear(tree.comX[0], weightedX / totalMass, "Incorrect root comX");

        expectRelativeNear(tree.comY[0], weightedY / totalMass, "Incorrect root comY");

        expectRelativeNear(tree.comZ[0], weightedZ / totalMass, "Incorrect root comZ");
    }
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
        
         /*
        * sortedIndices maps:
        *
        * Morton-sorted position
        *          ↓
        * original particle index
        */
        std::vector<std::uint32_t> sortedIndices(
            static_cast<std::size_t>(
                nParticles));

        for (int i = 0;
            i < nParticles;
            ++i)
        {
            sortedIndices[i] =
                static_cast<std::uint32_t>(i);
        }

        /*
        * Make the permutation deliberately non-identity
        * whenever there is more than one particle.
        */
        if (nParticles > 1)
        {
            std::rotate(
                sortedIndices.begin(),
                sortedIndices.begin() + 1,
                sortedIndices.end());
        }

        std::vector<double> mass(
            static_cast<std::size_t>(
                nParticles));

        std::vector<double> positionX(
            static_cast<std::size_t>(
                nParticles));

        std::vector<double> positionY(
            static_cast<std::size_t>(
                nParticles));

        std::vector<double> positionZ(
            static_cast<std::size_t>(
                nParticles));

        for (int i = 0;
            i < nParticles;
            ++i)
        {
            mass[i] =
                static_cast<double>(i + 1);

            positionX[i] =
                static_cast<double>(i);

            positionY[i] =
                static_cast<double>(2 * i);

            positionZ[i] =
                static_cast<double>(-i);
        }

        DeviceArray<std::uint32_t>
            deviceSortedIndices(
                sortedIndices);

        DeviceArray<double>
            deviceMass(
                mass);

        DeviceArray<double>
            deviceX(
                positionX);

        DeviceArray<double>
            deviceY(
                positionY);

        DeviceArray<double>
            deviceZ(
                positionZ);

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

        allocateOctreePhysicalData(
            octree);

        computeOctreeMassAndCenterOfMass(
            octree,
            deviceSortedIndices.get(),
            deviceMass.get(),
            deviceX.get(),
            deviceY.get(),
            deviceZ.get());

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

        validatePhysics(
            host,
            octree.nNodes,
            nParticles,
            sortedIndices,
            mass,
            positionX,
            positionY,
            positionZ);

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

class OctreePhysicsTest : public CudaTest
{
};

TEST_F(OctreePhysicsTest, ComputesSingleOccupiedCell)
{
    runCase("single occupied cell", {0u});
}

TEST_F(OctreePhysicsTest, ComputesDuplicateParticlesInOneCell)
{
    runCase(
        "duplicate particles in one cell",
        {7u, 7u, 7u, 7u});
}

TEST_F(OctreePhysicsTest, ComputesTwoLevelOneOctants)
{
    runCase(
        "two level-one octants",
        {0u, 1u << 29});
}

TEST_F(OctreePhysicsTest, ComputesSharedLevelOneCell)
{
    runCase(
        "shared level-one cell",
        {0u, 1u << 26});
}

TEST_F(OctreePhysicsTest, ComputesMixedOccupiedCells)
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

TEST_F(OctreePhysicsTest, ComputesDeterministicGeneratedInput)
{
    testRandom();
}
