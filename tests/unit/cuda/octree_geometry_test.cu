#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cuda_test.hpp"
#include "globalbounding.hpp"
#include "octree_builder.hpp"
#include "octree_geometry.hpp"
#include "radix_to_octree.hpp"

namespace
{

class DeviceCoordinates
{
public:
    DeviceCoordinates(
        const std::vector<double>& x,
        const std::vector<double>& y,
        const std::vector<double>& z)
        : x_(x.begin(), x.end()),
          y_(y.begin(), y.end()),
          z_(z.begin(), z.end())
    {
        if (x.empty() ||
            x.size() != y.size() ||
            x.size() != z.size())
        {
            throw std::invalid_argument(
                "Test coordinates must have equal non-zero sizes");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return x_.size();
    }

    [[nodiscard]] const double* x() const noexcept
    {
        return thrust::raw_pointer_cast(x_.data());
    }

    [[nodiscard]] const double* y() const noexcept
    {
        return thrust::raw_pointer_cast(y_.data());
    }

    [[nodiscard]] const double* z() const noexcept
    {
        return thrust::raw_pointer_cast(z_.data());
    }

private:
    thrust::device_vector<double> x_;
    thrust::device_vector<double> y_;
    thrust::device_vector<double> z_;
};


template <typename T>
std::vector<T> copyFromDevice(
    const T* source,
    std::size_t count)
{
    std::vector<T> result(count);

    const cudaError_t status =
        cudaMemcpy(
            result.data(),
            source,
            count * sizeof(T),
            cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(status));
    }

    return result;
}


void copyTopologyToDevice(
    Octree& octree,
    const std::vector<int>& levels,
    const std::vector<std::uint32_t>& prefixes)
{
    ASSERT_EQ(levels.size(), prefixes.size());

    allocateOctreeTopology(
        octree,
        static_cast<int>(levels.size()));

    ASSERT_EQ(
        cudaMemcpy(
            octree.level,
            levels.data(),
            levels.size() * sizeof(int),
            cudaMemcpyHostToDevice),
        cudaSuccess);

    ASSERT_EQ(
        cudaMemcpy(
            octree.prefix,
            prefixes.data(),
            prefixes.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice),
        cudaSuccess);

    octree.root = 0;
}


class OctreeGeometryTest : public CudaTest
{
protected:
    Octree octree;

    void TearDown() override
    {
        freeOctree(octree);
    }
};


TEST_F(
    OctreeGeometryTest,
    ComputesRootGeometryFromBoundingBox)
{
    const DeviceCoordinates coordinates(
        {-10.0, 10.0},
        {-10.0, 10.0},
        {-10.0, 10.0});

    const Bbox box(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    copyTopologyToDevice(
        octree,
        {0},
        {0u});

    allocateOctreeGeometry(octree);
    computeOctreeGeometry(octree, box);

    const auto centerX =
        copyFromDevice(octree.centerX, 1);

    const auto centerY =
        copyFromDevice(octree.centerY, 1);

    const auto centerZ =
        copyFromDevice(octree.centerZ, 1);

    const auto halfSize =
        copyFromDevice(octree.halfSize, 1);

    // Input range is 20.
    // Bbox adds 10%, therefore side = 22.
    EXPECT_DOUBLE_EQ(centerX[0], 0.0);
    EXPECT_DOUBLE_EQ(centerY[0], 0.0);
    EXPECT_DOUBLE_EQ(centerZ[0], 0.0);
    EXPECT_DOUBLE_EQ(halfSize[0], 11.0);
}


TEST_F(
    OctreeGeometryTest,
    ComputesOppositeLevelOneOctants)
{
    const DeviceCoordinates coordinates(
        {-10.0, 10.0},
        {-10.0, 10.0},
        {-10.0, 10.0});

    const Bbox box(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    constexpr std::uint32_t octant000 =
        0u << 27;

    constexpr std::uint32_t octant111 =
        7u << 27;

    copyTopologyToDevice(
        octree,
        {0, 1, 1},
        {0u, octant000, octant111});

    allocateOctreeGeometry(octree);
    computeOctreeGeometry(octree, box);

    const auto centerX =
        copyFromDevice(
            octree.centerX,
            octree.nNodes);

    const auto centerY =
        copyFromDevice(
            octree.centerY,
            octree.nNodes);

    const auto centerZ =
        copyFromDevice(
            octree.centerZ,
            octree.nNodes);

    const auto halfSize =
        copyFromDevice(
            octree.halfSize,
            octree.nNodes);

    // Root half-size = 11.
    // Level-one half-size = 5.5.
    //
    // octant 000 -> (-x, -y, -z)
    EXPECT_DOUBLE_EQ(centerX[1], -5.5);
    EXPECT_DOUBLE_EQ(centerY[1], -5.5);
    EXPECT_DOUBLE_EQ(centerZ[1], -5.5);
    EXPECT_DOUBLE_EQ(halfSize[1], 5.5);

    // octant 111 -> (+x, +y, +z)
    EXPECT_DOUBLE_EQ(centerX[2], 5.5);
    EXPECT_DOUBLE_EQ(centerY[2], 5.5);
    EXPECT_DOUBLE_EQ(centerZ[2], 5.5);
    EXPECT_DOUBLE_EQ(halfSize[2], 5.5);
}


TEST_F(
    OctreeGeometryTest,
    ReconstructsDeepNodeFromMortonPrefix)
{
    const DeviceCoordinates coordinates(
        {-10.0, 10.0},
        {-10.0, 10.0},
        {-10.0, 10.0});

    const Bbox box(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    //
    // Path:
    //
    // level 1: octant 5 = 101
    // level 2: octant 2 = 010
    // level 3: octant 7 = 111
    //
    constexpr std::uint32_t prefix =
        (5u << 27) |
        (2u << 24) |
        (7u << 21);

    copyTopologyToDevice(
        octree,
        {0, 3},
        {0u, prefix});

    allocateOctreeGeometry(octree);
    computeOctreeGeometry(octree, box);

    const auto centerX =
        copyFromDevice(octree.centerX, 2);

    const auto centerY =
        copyFromDevice(octree.centerY, 2);

    const auto centerZ =
        copyFromDevice(octree.centerZ, 2);

    const auto halfSize =
        copyFromDevice(octree.halfSize, 2);

    //
    // Root:
    //   half = 11
    //
    // octant 5 = 101:
    //   half = 5.5
    //   center = (5.5, -5.5, 5.5)
    //
    // octant 2 = 010:
    //   half = 2.75
    //   center = (2.75, -2.75, 2.75)
    //
    // octant 7 = 111:
    //   half = 1.375
    //   center = (4.125, -1.375, 4.125)
    //
    EXPECT_DOUBLE_EQ(centerX[1], 4.125);
    EXPECT_DOUBLE_EQ(centerY[1], -1.375);
    EXPECT_DOUBLE_EQ(centerZ[1], 4.125);
    EXPECT_DOUBLE_EQ(halfSize[1], 1.375);
}


TEST_F(
    OctreeGeometryTest,
    RejectsInvalidNodeLevel)
{
    const DeviceCoordinates coordinates(
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0});

    const Bbox box(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    copyTopologyToDevice(
        octree,
        {0, kOctreeMaxLevel + 1},
        {0u, 0u});

    allocateOctreeGeometry(octree);

    EXPECT_THROW(
        computeOctreeGeometry(
            octree,
            box),
        std::runtime_error);
}


TEST_F(
    OctreeGeometryTest,
    RejectsPrefixWithBitsBelowNodeLevel)
{
    const DeviceCoordinates coordinates(
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0});

    const Bbox box(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    //
    // Level 1 may only use bits 27..29.
    // Bit zero must therefore be zero.
    //
    copyTopologyToDevice(
        octree,
        {0, 1},
        {0u, 1u});

    allocateOctreeGeometry(octree);

    EXPECT_THROW(
        computeOctreeGeometry(
            octree,
            box),
        std::runtime_error);
}


TEST_F(
    OctreeGeometryTest,
    RequiresAllocatedGeometry)
{
    const DeviceCoordinates coordinates(
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0});

    const Bbox box(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    copyTopologyToDevice(
        octree,
        {0},
        {0u});

    EXPECT_THROW(
        computeOctreeGeometry(
            octree,
            box),
        std::logic_error);
}


TEST_F(
    OctreeGeometryTest,
    RejectsInvalidBoundingBox)
{
    const Bbox invalidBox;

    copyTopologyToDevice(
        octree,
        {0},
        {0u});

    allocateOctreeGeometry(octree);

    EXPECT_THROW(
        computeOctreeGeometry(
            octree,
            invalidBox),
        std::invalid_argument);
}

} // namespace