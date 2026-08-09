#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tree_builder.hpp"

namespace {

void cudaCheck(cudaError_t error, const char* expression, const char* file, int line)
{
    if (error == cudaSuccess)
        return;

    std::ostringstream message;
    message << file << ':' << line << ": " << expression
            << " failed: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __FILE__, __LINE__)

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <typename T>
class DeviceArray
{
public:
    explicit DeviceArray(const std::vector<T>& host) : size_(host.size())
    {
        if (size_ == 0)
            return;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
        try
        {
            CUDA_CHECK(cudaMemcpy(data_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
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
    DeviceArray& operator=(const DeviceArray&) = delete;

    const T* get() const noexcept { return data_; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

class TreeOwner
{
public:
    explicit TreeOwner(int n)
    {
        allocateTree(tree_, n);
    }

    ~TreeOwner()
    {
        freeTree(tree_);
    }

    TreeOwner(const TreeOwner&) = delete;
    TreeOwner& operator=(const TreeOwner&) = delete;

    Tree& get() noexcept { return tree_; }

private:
    Tree tree_{};
};

struct HostTree
{
    std::vector<int> left;
    std::vector<int> right;
    std::vector<int> parent;
    std::vector<int> rangeFirst;
    std::vector<int> rangeLast;
    std::vector<int> prefixLength;
    std::vector<int> visitCount;
    std::vector<float> mass;
    std::vector<float> comX;
    std::vector<float> comY;
    std::vector<float> comZ;
};

HostTree copyTreeToHost(const Tree& tree, int n, bool copyPhysicalData)
{
    HostTree host;
    const std::size_t internalCount = static_cast<std::size_t>(n - 1);
    const std::size_t totalCount = static_cast<std::size_t>(2 * n - 1);

    host.parent.resize(totalCount);
    CUDA_CHECK(cudaMemcpy(host.parent.data(), tree.parent,
                          totalCount * sizeof(int), cudaMemcpyDeviceToHost));

    if (n > 1)
    {
        host.left.resize(internalCount);
        host.right.resize(internalCount);
        CUDA_CHECK(cudaMemcpy(host.left.data(), tree.left,
                              internalCount * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.right.data(), tree.right,
                              internalCount * sizeof(int), cudaMemcpyDeviceToHost));

        host.rangeFirst.resize(internalCount);
        host.rangeLast.resize(internalCount);
        host.prefixLength.resize(internalCount);

        CUDA_CHECK(cudaMemcpy(host.rangeFirst.data(), tree.rangeFirst,
                              internalCount * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.rangeLast.data(), tree.rangeLast,
                              internalCount * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.prefixLength.data(), tree.prefixLength,
                              internalCount * sizeof(int), cudaMemcpyDeviceToHost));
    }

    if (copyPhysicalData)
    {
        host.mass.resize(totalCount);
        host.comX.resize(totalCount);
        host.comY.resize(totalCount);
        host.comZ.resize(totalCount);

        CUDA_CHECK(cudaMemcpy(host.mass.data(), tree.mass,
                              totalCount * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.comX.data(), tree.comX,
                              totalCount * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.comY.data(), tree.comY,
                              totalCount * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host.comZ.data(), tree.comZ,
                              totalCount * sizeof(float), cudaMemcpyDeviceToHost));

        if (n > 1)
        {
            host.visitCount.resize(internalCount);
            CUDA_CHECK(cudaMemcpy(host.visitCount.data(), tree.visitCount,
                                  internalCount * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    return host;
}

void requireSorted(const std::vector<std::uint32_t>& keys)
{
    require(std::is_sorted(keys.begin(), keys.end()),
            "The test input Morton keys are not sorted");
}

void requirePermutation(const std::vector<std::uint32_t>& indices, int n)
{
    require(static_cast<int>(indices.size()) == n,
            "sortedIndices has the wrong size");

    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    for (std::uint32_t index : indices)
    {
        require(index < static_cast<std::uint32_t>(n),
                "sortedIndices contains an out-of-range particle index");
        ++seen[index];
    }

    for (int count : seen)
        require(count == 1, "sortedIndices is not a permutation of [0, N)");
}

int countLeadingZeros32(std::uint32_t value)
{
    require(value != 0u, "countLeadingZeros32 requires a non-zero value");

    int count = 0;
    while ((value & 0x80000000u) == 0u)
    {
        ++count;
        value <<= 1;
    }
    return count;
}

int hostLongestCommonPrefix(const std::vector<std::uint32_t>& keys,
                            int i,
                            int j)
{
    const int n = static_cast<int>(keys.size());

    if (j < 0 || j >= n)
        return -1;

    if (i == j)
        return 64;

    const std::uint32_t keyDifference = keys[i] ^ keys[j];
    if (keyDifference != 0u)
        return countLeadingZeros32(keyDifference);

    const std::uint32_t indexDifference =
        static_cast<std::uint32_t>(i ^ j);
    return 32 + countLeadingZeros32(indexDifference);
}

struct LeafRange
{
    int first = -1;
    int last = -1;
    int count = 0;
};

int validateTopology(const HostTree& tree,
                     const std::vector<std::uint32_t>& keys)
{
    const int n = static_cast<int>(keys.size());
    const int totalNodes = 2 * n - 1;

    require(static_cast<int>(tree.parent.size()) == totalNodes,
            "parent has the wrong number of elements");

    if (n == 1)
    {
        require(tree.left.empty() && tree.right.empty(),
                "A one-body tree must have no internal child arrays");
        require(tree.rangeFirst.empty() && tree.rangeLast.empty() &&
                    tree.prefixLength.empty(),
                "A one-body tree must have no internal metadata arrays");
        require(tree.parent[0] == -1,
                "The single leaf must also be the root");
        return 0;
    }

    require(static_cast<int>(tree.left.size()) == n - 1,
            "left has the wrong number of elements");
    require(static_cast<int>(tree.right.size()) == n - 1,
            "right has the wrong number of elements");
    require(static_cast<int>(tree.rangeFirst.size()) == n - 1,
            "rangeFirst has the wrong number of elements");
    require(static_cast<int>(tree.rangeLast.size()) == n - 1,
            "rangeLast has the wrong number of elements");
    require(static_cast<int>(tree.prefixLength.size()) == n - 1,
            "prefixLength has the wrong number of elements");

    std::vector<int> incoming(static_cast<std::size_t>(totalNodes), 0);

    for (int internalIndex = 0; internalIndex < n - 1; ++internalIndex)
    {
        const int node = n + internalIndex;
        const int leftChild = tree.left[internalIndex];
        const int rightChild = tree.right[internalIndex];

        require(leftChild >= 0 && leftChild < totalNodes,
                "left child is outside [0, 2N-1)");
        require(rightChild >= 0 && rightChild < totalNodes,
                "right child is outside [0, 2N-1)");
        require(leftChild != rightChild,
                "An internal node has the same left and right child");
        require(tree.parent[leftChild] == node,
                "parent[leftChild] does not point back to its owner");
        require(tree.parent[rightChild] == node,
                "parent[rightChild] does not point back to its owner");

        ++incoming[leftChild];
        ++incoming[rightChild];
    }

    std::vector<int> roots;
    for (int node = n; node < totalNodes; ++node)
    {
        if (tree.parent[node] == -1)
            roots.push_back(node);
    }

    require(roots.size() == 1,
            "There must be exactly one internal node with parent == -1");

    const int root = roots.front();
    require(root == n,
            "Karras indexing should place the root at global node N");

    for (int node = 0; node < totalNodes; ++node)
    {
        const int expectedIncoming = (node == root) ? 0 : 1;
        require(incoming[node] == expectedIncoming,
                "A node has an invalid number of incoming child references");

        if (node != root)
        {
            require(tree.parent[node] >= n && tree.parent[node] < totalNodes,
                    "A non-root node has an invalid parent ID");
        }
    }

    std::vector<int> state(static_cast<std::size_t>(totalNodes), 0);
    int visitedNodes = 0;

    std::function<LeafRange(int)> visit = [&](int node) -> LeafRange {
        require(node >= 0 && node < totalNodes,
                "Traversal reached an invalid node ID");
        require(state[node] == 0,
                "The topology contains a cycle or a node reachable twice");

        state[node] = 1;
        ++visitedNodes;

        if (node < n)
        {
            state[node] = 2;
            return LeafRange{node, node, 1};
        }

        const int internalIndex = node - n;
        const LeafRange leftRange = visit(tree.left[internalIndex]);
        const LeafRange rightRange = visit(tree.right[internalIndex]);

        require(leftRange.last + 1 == rightRange.first,
                "Left and right subtrees are not adjacent Morton-order ranges");

        const LeafRange result{
            leftRange.first,
            rightRange.last,
            leftRange.count + rightRange.count
        };

        require(result.count == result.last - result.first + 1,
                "An internal node does not cover a contiguous leaf range");

        require(tree.rangeFirst[internalIndex] == result.first,
                "rangeFirst does not match the subtree topology");
        require(tree.rangeLast[internalIndex] == result.last,
                "rangeLast does not match the subtree topology");

        const int expectedPrefix =
            hostLongestCommonPrefix(keys, result.first, result.last);
        require(tree.prefixLength[internalIndex] == expectedPrefix,
                "prefixLength does not match the node Morton-key range");

        state[node] = 2;
        return result;
    };

    const LeafRange rootRange = visit(root);
    require(rootRange.first == 0 && rootRange.last == n - 1 && rootRange.count == n,
            "The root does not cover all Morton-order leaves");
    require(visitedNodes == totalNodes,
            "Some nodes are disconnected from the root");

    return root;
}

bool almostEqual(float actual, float expected)
{
    const float scale = std::max({1.0f, std::fabs(actual), std::fabs(expected)});
    return std::fabs(actual - expected) <= 2.0e-5f * scale;
}

void requireAlmostEqual(float actual, float expected, const std::string& label)
{
    if (almostEqual(actual, expected))
        return;

    std::ostringstream message;
    message << std::setprecision(9) << label << ": expected " << expected
            << ", got " << actual;
    throw std::runtime_error(message.str());
}

template <typename T>
void requireVectorEqual(const std::vector<T>& actual,
                        const std::vector<T>& expected,
                        const std::string& label)
{
    require(actual.size() == expected.size(), label + " has the wrong size");

    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        if (actual[i] == expected[i])
            continue;

        std::ostringstream message;
        message << label << '[' << i << "]: expected " << expected[i]
                << ", got " << actual[i];
        throw std::runtime_error(message.str());
    }
}

HostTree buildAndCopyTopology(const std::vector<std::uint32_t>& keys)
{
    require(!keys.empty(), "At least one Morton key is required");
    requireSorted(keys);

    const int n = static_cast<int>(keys.size());
    DeviceArray<std::uint32_t> deviceKeys(keys);
    TreeOwner treeOwner(n);

    buildTree(treeOwner.get(), deviceKeys.get(), n);
    HostTree hostTree = copyTreeToHost(treeOwner.get(), n, false);
    validateTopology(hostTree, keys);
    return hostTree;
}

struct Aggregate
{
    float mass = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

void verifyCenterOfMass(const HostTree& tree,
                        int n,
                        const std::vector<std::uint32_t>& sortedIndices,
                        const std::vector<float>& particleMass,
                        const std::vector<float>& positionX,
                        const std::vector<float>& positionY,
                        const std::vector<float>& positionZ)
{
    const int totalNodes = 2 * n - 1;
    require(static_cast<int>(tree.mass.size()) == totalNodes,
            "Physical arrays were not copied");

    for (int leaf = 0; leaf < n; ++leaf)
    {
        const std::uint32_t particle = sortedIndices[leaf];
        requireAlmostEqual(tree.mass[leaf], particleMass[particle],
                           "leaf mass");
        requireAlmostEqual(tree.comX[leaf], positionX[particle],
                           "leaf comX");
        requireAlmostEqual(tree.comY[leaf], positionY[particle],
                           "leaf comY");
        requireAlmostEqual(tree.comZ[leaf], positionZ[particle],
                           "leaf comZ");
    }

    std::function<Aggregate(int)> expectedForNode = [&](int node) -> Aggregate {
        if (node < n)
        {
            const std::uint32_t particle = sortedIndices[node];
            return Aggregate{
                particleMass[particle],
                positionX[particle],
                positionY[particle],
                positionZ[particle]
            };
        }

        const int internalIndex = node - n;
        const Aggregate left = expectedForNode(tree.left[internalIndex]);
        const Aggregate right = expectedForNode(tree.right[internalIndex]);
        const float totalMass = left.mass + right.mass;

        Aggregate result;
        result.mass = totalMass;
        if (totalMass > 0.0f)
        {
            result.x = (left.mass * left.x + right.mass * right.x) / totalMass;
            result.y = (left.mass * left.y + right.mass * right.y) / totalMass;
            result.z = (left.mass * left.z + right.mass * right.z) / totalMass;
        }
        return result;
    };

    for (int node = n; node < totalNodes; ++node)
    {
        const Aggregate expected = expectedForNode(node);
        requireAlmostEqual(tree.mass[node], expected.mass,
                           "internal mass at node " + std::to_string(node));
        requireAlmostEqual(tree.comX[node], expected.x,
                           "internal comX at node " + std::to_string(node));
        requireAlmostEqual(tree.comY[node], expected.y,
                           "internal comY at node " + std::to_string(node));
        requireAlmostEqual(tree.comZ[node], expected.z,
                           "internal comZ at node " + std::to_string(node));
    }

    if (n > 1)
    {
        require(static_cast<int>(tree.visitCount.size()) == n - 1,
                "visitCount has the wrong size");
        for (int count : tree.visitCount)
            require(count == 2, "Every internal node must be visited by exactly two children");
    }
}

void runEndToEndCase(const std::string& name,
                     const std::vector<std::uint32_t>& keys,
                     const std::vector<std::uint32_t>& sortedIndices,
                     const std::vector<float>& particleMass,
                     const std::vector<float>& positionX,
                     const std::vector<float>& positionY,
                     const std::vector<float>& positionZ)
{
    require(!keys.empty(), name + ": empty Morton-key vector");
    requireSorted(keys);

    const int n = static_cast<int>(keys.size());
    requirePermutation(sortedIndices, n);
    require(static_cast<int>(particleMass.size()) == n, name + ": wrong mass size");
    require(static_cast<int>(positionX.size()) == n, name + ": wrong X size");
    require(static_cast<int>(positionY.size()) == n, name + ": wrong Y size");
    require(static_cast<int>(positionZ.size()) == n, name + ": wrong Z size");

    DeviceArray<std::uint32_t> deviceKeys(keys);
    DeviceArray<std::uint32_t> deviceIndices(sortedIndices);
    DeviceArray<float> deviceMass(particleMass);
    DeviceArray<float> deviceX(positionX);
    DeviceArray<float> deviceY(positionY);
    DeviceArray<float> deviceZ(positionZ);
    TreeOwner treeOwner(n);

    buildTree(treeOwner.get(), deviceKeys.get(), n);
    computeCenterOfMass(treeOwner.get(), deviceIndices.get(), deviceMass.get(),
                        deviceX.get(), deviceY.get(), deviceZ.get(), n);

    HostTree hostTree = copyTreeToHost(treeOwner.get(), n, true);
    validateTopology(hostTree, keys);
    verifyCenterOfMass(hostTree, n, sortedIndices, particleMass,
                       positionX, positionY, positionZ);

    std::cout << "[PASS] " << name << '\n';
}

void testSingleLeaf()
{
    runEndToEndCase(
        "single leaf",
        {42u},
        {0u},
        {2.5f},
        {1.0f},
        {-3.0f},
        {7.0f});
}

void testTwoLeavesExactTopology()
{
    const HostTree tree = buildAndCopyTopology({0x00000000u, 0x80000000u});
    requireVectorEqual(tree.left, {0}, "two-leaf left");
    requireVectorEqual(tree.right, {1}, "two-leaf right");
    requireVectorEqual(tree.parent, {2, 2, -1}, "two-leaf parent");
    requireVectorEqual(tree.rangeFirst, {0}, "two-leaf rangeFirst");
    requireVectorEqual(tree.rangeLast, {1}, "two-leaf rangeLast");
    requireVectorEqual(tree.prefixLength, {0}, "two-leaf prefixLength");
    std::cout << "[PASS] exact topology: two leaves\n";
}

void testBalancedFourExactTopology()
{
    const HostTree tree = buildAndCopyTopology({
        0x00000000u,
        0x40000000u,
        0x80000000u,
        0xC0000000u
    });

    requireVectorEqual(tree.left, {5, 0, 2}, "balanced-four left");
    requireVectorEqual(tree.right, {6, 1, 3}, "balanced-four right");
    requireVectorEqual(tree.parent, {5, 5, 6, 6, -1, 4, 4},
                       "balanced-four parent");
    requireVectorEqual(tree.rangeFirst, {0, 0, 2},
                       "balanced-four rangeFirst");
    requireVectorEqual(tree.rangeLast, {3, 1, 3},
                       "balanced-four rangeLast");
    requireVectorEqual(tree.prefixLength, {0, 1, 1},
                       "balanced-four prefixLength");
    std::cout << "[PASS] exact topology: balanced four leaves\n";
}

void testPermutationAndCenterOfMass()
{
    runEndToEndCase(
        "sorted-index permutation and center of mass",
        {0x00000000u, 0x40000000u, 0x80000000u, 0xC0000000u},
        {2u, 0u, 3u, 1u},
        {1.0f, 2.0f, 3.0f, 4.0f},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {-1.0f, -2.0f, -3.0f, -4.0f},
        {5.0f, 15.0f, 25.0f, 35.0f});
}

void testDuplicateMortonKeys()
{
    runEndToEndCase(
        "duplicate Morton keys",
        {7u, 7u, 7u, 7u, 7u, 7u},
        {5u, 2u, 4u, 0u, 3u, 1u},
        {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f},
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
        {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f},
        {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f});
}

void testRandomCases()
{
    std::mt19937 generator(0xC0FFEEu);
    const std::vector<int> sizes{3, 5, 7, 16, 31, 64, 257};

    for (int n : sizes)
    {
        std::vector<std::uint32_t> keys(static_cast<std::size_t>(n));
        for (std::uint32_t& key : keys)
            key = generator();
        std::sort(keys.begin(), keys.end());

        // Deliberately retain some equal Morton codes while preserving order.
        for (int i = 5; i < n; i += 11)
            keys[i] = keys[i - 1];

        std::vector<std::uint32_t> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), 0u);
        std::shuffle(indices.begin(), indices.end(), generator);

        std::vector<float> mass(static_cast<std::size_t>(n));
        std::vector<float> x(static_cast<std::size_t>(n));
        std::vector<float> y(static_cast<std::size_t>(n));
        std::vector<float> z(static_cast<std::size_t>(n));

        for (int i = 0; i < n; ++i)
        {
            mass[i] = 0.5f + static_cast<float>((i % 13) + 1) * 0.25f;
            x[i] = static_cast<float>(i) * 0.75f - 10.0f;
            y[i] = static_cast<float>((i * i) % 29) - 14.0f;
            z[i] = static_cast<float>((i * 7) % 31) * 0.5f;
        }

        runEndToEndCase("random N=" + std::to_string(n), keys, indices,
                        mass, x, y, z);
    }
}

} // namespace

int runTreeBuilderTestSuite()
{
    try
    {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        require(deviceCount > 0, "No CUDA-capable GPU is available");

        testSingleLeaf();
        testTwoLeavesExactTopology();
        testBalancedFourExactTopology();
        testPermutationAndCenterOfMass();
        testDuplicateMortonKeys();
        testRandomCases();

        std::cout << "\nAll tree-builder tests passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "\n[FAIL] " << error.what() << '\n';
        return 1;
    }
}

TEST(TreeBuilderCudaTest, CompleteSuite)
{
    EXPECT_EQ(runTreeBuilderTestSuite(), 0);
}
