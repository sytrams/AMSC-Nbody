#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

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

void require(
    bool condition,
    const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(
    float actual,
    float expected,
    const std::string& message)
{
    constexpr float tolerance = 1.0e-5f;

    const float scale =
        std::max(
            1.0f,
            std::max(
                std::abs(actual),
                std::abs(expected)));

    if (std::abs(actual - expected) >
        tolerance * scale)
    {
        throw std::runtime_error(
            message +
            ": expected " +
            std::to_string(expected) +
            ", got " +
            std::to_string(actual));
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

    std::vector<float> mass;
    std::vector<float> comX;
    std::vector<float> comY;
    std::vector<float> comZ;

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
            n * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy octree mass");

    checkCuda(
        cudaMemcpy(
            host.comX.data(),
            octree.comX,
            n * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy octree comX");

    checkCuda(
        cudaMemcpy(
            host.comY.data(),
            octree.comY,
            n * sizeof(float),
            cudaMemcpyDeviceToHost),
        "copy octree comY");

    checkCuda(
        cudaMemcpy(
            host.comZ.data(),
            octree.comZ,
            n * sizeof(float),
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
    require(
        nNodes > 0,
        "Octree contains no nodes");

    require(
        tree.level[0] == 0,
        "Root level must be zero");

    require(
        tree.prefix[0] == 0u,
        "Root prefix must be zero");

    require(
        tree.parent[0] == -1,
        "Root parent must be -1");

    std::vector<int> incoming(
        static_cast<std::size_t>(nNodes),
        0);

    for (int node = 0;
         node < nNodes;
         ++node)
    {
        require(
            tree.level[node] >= 0 &&
            tree.level[node] <=
                kOctreeMaxLevel,
            "Invalid octree level");

        require(
            tree.prefix[node] ==
                prefixAtLevel(
                    tree.prefix[node],
                    tree.level[node]),
            "Octree prefix contains bits "
            "below its level");

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

            require(
                child > 0 &&
                child < nNodes,
                "Child ID outside octree");

            require(
                tree.parent[child] ==
                    node,
                "Child parent pointer "
                "does not point back");

            require(
                tree.level[child] ==
                    tree.level[node] + 1,
                "Octree edge skips a level");

            require(
                octantAtLevel(
                    tree.prefix[child],
                    tree.level[child]) ==
                    octant,
                "Child stored in wrong octant");

            require(
                prefixAtLevel(
                    tree.prefix[child],
                    tree.level[node]) ==
                    tree.prefix[node],
                "Child prefix does not extend "
                "parent prefix");

            ++incoming[child];
        }
    }

    require(
        incoming[0] == 0,
        "Root has an incoming edge");

    for (int node = 1;
         node < nNodes;
         ++node)
    {
        require(
            incoming[node] == 1,
            "Non-root node does not have "
            "exactly one incoming edge");
    }

    std::vector<int> visited(
        static_cast<std::size_t>(nNodes),
        0);

    int visitedCount = 0;

    std::function<void(int)> visit =
        [&](int node)
    {
        require(
            visited[node] == 0,
            "Cycle or repeated octree node");

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

    require(
        visitedCount == nNodes,
        "Some octree nodes are unreachable "
        "from the root");
}

void validatePhysics(
    const HostOctree& tree,
    int nNodes,
    int nParticles,
    const std::vector<std::uint32_t>& sortedIndices,
    const std::vector<float>& particleMass,
    const std::vector<float>& positionX,
    const std::vector<float>& positionY,
    const std::vector<float>& positionZ)
{
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

        require(
            tree.childCount[node] == 0,
            "Occupied leaf has children");

        require(
            tree.pendingChildren[node] == 0,
            "Occupied leaf has pending children");

        const int first =
            tree.firstParticle[node];

        const int count =
            tree.particleCount[node];

        require(
            first >= 0 &&
            first + count <= nParticles,
            "Invalid occupied-leaf particle range");

        float expectedMass = 0.0f;
        float weightedX = 0.0f;
        float weightedY = 0.0f;
        float weightedZ = 0.0f;

        for (int local = 0;
             local < count;
             ++local)
        {
            const int sortedPosition =
                first + local;

            const std::uint32_t particle =
                sortedIndices[sortedPosition];

            require(
                particle <
                    static_cast<std::uint32_t>(
                        nParticles),
                "Invalid sorted particle index");

            const float mass =
                particleMass[particle];

            expectedMass += mass;

            weightedX +=
                mass * positionX[particle];

            weightedY +=
                mass * positionY[particle];

            weightedZ +=
                mass * positionZ[particle];
        }

        requireNear(
            tree.mass[node],
            expectedMass,
            "Incorrect leaf mass");

        if (expectedMass > 0.0f)
        {
            requireNear(
                tree.comX[node],
                weightedX / expectedMass,
                "Incorrect leaf comX");

            requireNear(
                tree.comY[node],
                weightedY / expectedMass,
                "Incorrect leaf comY");

            requireNear(
                tree.comZ[node],
                weightedZ / expectedMass,
                "Incorrect leaf comZ");
        }
        else
        {
            requireNear(
                tree.comX[node],
                0.0f,
                "Zero-mass leaf comX");

            requireNear(
                tree.comY[node],
                0.0f,
                "Zero-mass leaf comY");

            requireNear(
                tree.comZ[node],
                0.0f,
                "Zero-mass leaf comZ");
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

        float expectedMass = 0.0f;
        float weightedX = 0.0f;
        float weightedY = 0.0f;
        float weightedZ = 0.0f;

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

            const float childMass =
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

        require(
            tree.childCount[node] ==
                actualChildCount,
            "Incorrect childCount");

        if (actualChildCount == 0)
            continue;

        require(
            tree.pendingChildren[node] ==
                actualChildCount,
            "Internal node did not receive "
            "all child completions");

        requireNear(
            tree.mass[node],
            expectedMass,
            "Incorrect internal-node mass");

        if (expectedMass > 0.0f)
        {
            requireNear(
                tree.comX[node],
                weightedX / expectedMass,
                "Incorrect internal comX");

            requireNear(
                tree.comY[node],
                weightedY / expectedMass,
                "Incorrect internal comY");

            requireNear(
                tree.comZ[node],
                weightedZ / expectedMass,
                "Incorrect internal comZ");
        }
        else
        {
            requireNear(
                tree.comX[node],
                0.0f,
                "Zero-mass internal comX");

            requireNear(
                tree.comY[node],
                0.0f,
                "Zero-mass internal comY");

            requireNear(
                tree.comZ[node],
                0.0f,
                "Zero-mass internal comZ");
        }
    }

    /*
     * 3. Completely independent reference calculation
     *    for the root from ALL original particles.
     */
    float totalMass = 0.0f;
    float weightedX = 0.0f;
    float weightedY = 0.0f;
    float weightedZ = 0.0f;

    for (int particle = 0;
         particle < nParticles;
         ++particle)
    {
        const float mass =
            particleMass[particle];

        totalMass += mass;

        weightedX +=
            mass * positionX[particle];

        weightedY +=
            mass * positionY[particle];

        weightedZ +=
            mass * positionZ[particle];
    }

    requireNear(
        tree.mass[0],
        totalMass,
        "Incorrect root mass");

    if (totalMass > 0.0f)
    {
        requireNear(
            tree.comX[0],
            weightedX / totalMass,
            "Incorrect root comX");

        requireNear(
            tree.comY[0],
            weightedY / totalMass,
            "Incorrect root comY");

        requireNear(
            tree.comZ[0],
            weightedZ / totalMass,
            "Incorrect root comZ");
    }
}

void runCase(
    const std::string& name,
    const std::vector<std::uint32_t>& sortedKeys)
{
    require(
        !sortedKeys.empty(),
        name + ": empty input");

    require(
        std::is_sorted(
            sortedKeys.begin(),
            sortedKeys.end()),
        name + ": keys are not sorted");

    DeviceArray<std::uint32_t>
        deviceKeys(sortedKeys);

    MortonLeafGroups groups{};
    Tree radixTree{};
    RadixToOctreePlan plan{};
    Octree octree{};

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

        std::vector<float> mass(
            static_cast<std::size_t>(
                nParticles));

        std::vector<float> positionX(
            static_cast<std::size_t>(
                nParticles));

        std::vector<float> positionY(
            static_cast<std::size_t>(
                nParticles));

        std::vector<float> positionZ(
            static_cast<std::size_t>(
                nParticles));

        for (int i = 0;
            i < nParticles;
            ++i)
        {
            mass[i] =
                static_cast<float>(i + 1);

            positionX[i] =
                static_cast<float>(i);

            positionY[i] =
                static_cast<float>(2 * i);

            positionZ[i] =
                static_cast<float>(-i);
        }

        DeviceArray<std::uint32_t>
            deviceSortedIndices(
                sortedIndices);

        DeviceArray<float>
            deviceMass(
                mass);

        DeviceArray<float>
            deviceX(
                positionX);

        DeviceArray<float>
            deviceY(
                positionY);

        DeviceArray<float>
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

        require(
            octree.root == 0,
            name +
            ": root index is not zero");

        require(
            octree.nNodes ==
                plan.nOctreeNodes,
            name +
            ": wrong node count");

        require(
            octree.nLeaves ==
                groups.nGroups,
            name +
            ": wrong occupied leaf count");

        require(
            octree.nParticles ==
                nParticles,
            name +
            ": wrong particle count");

        const HostOctree host =
            copyOctree(octree);

        validateTopology(
            host,
            octree.nNodes);

        validatePhysics(
            host,
            octree.nNodes,
            nParticles,
            sortedIndices,
            mass,
            positionX,
            positionY,
            positionZ);

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
                groups.uniqueKeys,
                uniqueKeys.size() *
                    sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost),
            "copy unique keys");

        checkCuda(
            cudaMemcpy(
                firstParticle.data(),
                groups.firstParticle,
                firstParticle.size() *
                    sizeof(int),
                cudaMemcpyDeviceToHost),
            "copy group firstParticle");

        checkCuda(
            cudaMemcpy(
                particleCount.data(),
                groups.particleCount,
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
                require(
                    host.firstParticle[node] == -1,
                    name +
                    ": non-leaf contains particle "
                    "metadata");

                continue;
            }

            ++occupiedLeaves;

            require(
                host.level[node] ==
                    kOctreeMaxLevel,
                name +
                ": occupied leaf is not "
                "at level 10");

            bool matched = false;

            for (int group = 0;
                 group < groups.nGroups;
                 ++group)
            {
                if (host.prefix[node] !=
                    uniqueKeys[group])
                {
                    continue;
                }

                require(
                    host.firstParticle[node] ==
                        firstParticle[group],
                    name +
                    ": incorrect leaf offset");

                require(
                    host.particleCount[node] ==
                        particleCount[group],
                    name +
                    ": incorrect leaf count");

                matched = true;
                break;
            }

            require(
                matched,
                name +
                ": occupied leaf does not "
                "correspond to a Morton group");
        }

        require(
            occupiedLeaves ==
                groups.nGroups,
            name +
            ": not every Morton group "
            "has one occupied leaf");
    }
    catch (...)
    {
        freeOctree(octree);
        freeRadixToOctreePlan(plan);
        freeTree(radixTree);
        freeMortonLeafGroups(groups);
        throw;
    }

    freeOctree(octree);
    freeRadixToOctreePlan(plan);
    freeTree(radixTree);
    freeMortonLeafGroups(groups);

    std::cout
        << "[PASS] "
        << name
        << '\n';
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

int runOctreeMassCenterOfMassTestSuite()
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
            "No CUDA-capable GPU available");

        runCase(
            "single occupied cell",
            {0u});

        runCase(
            "duplicate particles in one cell",
            {
                7u,
                7u,
                7u,
                7u
            });

        runCase(
            "two level-one octants",
            {
                0u,
                1u << 29
            });

        runCase(
            "shared level-one cell",
            {
                0u,
                1u << 26
            });

        runCase(
            "mixed occupied cells",
            {
                0u,
                1u << 23,
                1u << 26,
                1u << 29,
                (1u << 29) |
                    (1u << 26)
            });

        testRandom();

        std::cout
            << "\nAll octree mass/CoM "
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

TEST(OctreeMassCenterOfMassCudaTest, CompleteSuite)
{
    EXPECT_EQ(runOctreeMassCenterOfMassTestSuite(), 0);
}
