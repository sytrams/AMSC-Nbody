#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_test.hpp"
#include "morton.hpp"

namespace {

void checkCuda(cudaError_t error, const char* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) +
            " failed: " +
            cudaGetErrorString(error));
    }
}

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
        if (x.empty() || x.size() != y.size() || x.size() != z.size())
        {
            throw std::invalid_argument(
                "Test coordinates must be non-empty and have equal sizes");
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

struct MortonSnapshot
{
    std::vector<MortonKey> keys;
    std::vector<ParticleIndex> indices;
};

MortonSnapshot copyMortonToHost(const MortonKeys& morton)
{
    MortonSnapshot snapshot{
        std::vector<MortonKey>(morton.size()),
        std::vector<ParticleIndex>(morton.size())};

    const std::size_t keyBytes =
        snapshot.keys.size() * sizeof(MortonKey);
    const std::size_t indexBytes =
        snapshot.indices.size() * sizeof(ParticleIndex);

    checkCuda(
        cudaMemcpy(
            snapshot.keys.data(),
            morton.keys_device_data(),
            keyBytes,
            cudaMemcpyDeviceToHost),
        "copy Morton keys to host");

    checkCuda(
        cudaMemcpy(
            snapshot.indices.data(),
            morton.indices_device_data(),
            indexBytes,
            cudaMemcpyDeviceToHost),
        "copy Morton indices to host");

    return snapshot;
}

std::uint32_t expandBitsHost(std::uint32_t value)
{
    std::uint32_t x = value & 0x000003FFu;

    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;

    return x;
}

MortonKey mortonKeyHost(
    std::uint32_t qx,
    std::uint32_t qy,
    std::uint32_t qz)
{
    return (expandBitsHost(qx) << 2) |
           (expandBitsHost(qy) << 1) |
           expandBitsHost(qz);
}

std::uint32_t quantiseHost(
    double coordinate,
    double centre,
    double side)
{
    constexpr std::uint32_t gridSize = 1u << 10;
    constexpr std::uint32_t maxQ = gridSize - 1u;

    const double normalised =
        (coordinate - (centre - side * 0.5)) / side;
    const double scaled =
        normalised * static_cast<double>(gridSize);
    const double clamped =
        std::clamp(scaled, 0.0, static_cast<double>(maxQ));

    return static_cast<std::uint32_t>(clamped);
}

MortonSnapshot referenceMorton(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z)
{
    if (x.empty() || x.size() != y.size() || x.size() != z.size())
    {
        throw std::invalid_argument(
            "Reference coordinates must be non-empty and have equal sizes");
    }

    const auto [xminIt, xmaxIt] =
        std::minmax_element(x.begin(), x.end());
    const auto [yminIt, ymaxIt] =
        std::minmax_element(y.begin(), y.end());
    const auto [zminIt, zmaxIt] =
        std::minmax_element(z.begin(), z.end());

    const double centreX = (*xminIt + *xmaxIt) * 0.5;
    const double centreY = (*yminIt + *ymaxIt) * 0.5;
    const double centreZ = (*zminIt + *zmaxIt) * 0.5;

    double side = std::max({
        *xmaxIt - *xminIt,
        *ymaxIt - *yminIt,
        *zmaxIt - *zminIt});

    if (side == 0.0)
    {
        side = 1.0;
    }

    std::vector<std::pair<MortonKey, ParticleIndex>> pairs;
    pairs.reserve(x.size());

    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const auto qx = quantiseHost(x[i], centreX, side);
        const auto qy = quantiseHost(y[i], centreY, side);
        const auto qz = quantiseHost(z[i], centreZ, side);

        pairs.emplace_back(
            mortonKeyHost(qx, qy, qz),
            static_cast<ParticleIndex>(i));
    }

    std::stable_sort(
        pairs.begin(),
        pairs.end(),
        [](const auto& lhs, const auto& rhs)
        {
            return lhs.first < rhs.first;
        });

    MortonSnapshot result;
    result.keys.reserve(pairs.size());
    result.indices.reserve(pairs.size());

    for (const auto& [key, index] : pairs)
    {
        result.keys.push_back(key);
        result.indices.push_back(index);
    }

    return result;
}

class MortonKeysTest : public CudaTest
{
};

TEST_F(MortonKeysTest, ComputesSortedKeysAndMatchingPermutation)
{
    const std::vector<double> x{0.8, -1.0, 0.1, 1.0, -0.4};
    const std::vector<double> y{-0.2, 0.9, 0.0, -1.0, 0.3};
    const std::vector<double> z{1.0, -0.7, 0.2, 0.4, -1.0};

    const DeviceCoordinates coordinates(x, y, z);
    const MortonKeys morton(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    const MortonSnapshot actual = copyMortonToHost(morton);
    const MortonSnapshot expected = referenceMorton(x, y, z);

    ASSERT_EQ(actual.keys.size(), x.size());
    ASSERT_EQ(actual.indices.size(), x.size());
    EXPECT_TRUE(std::is_sorted(actual.keys.begin(), actual.keys.end()));
    EXPECT_EQ(actual.keys, expected.keys);
    EXPECT_EQ(actual.indices, expected.indices);
}

TEST_F(MortonKeysTest, KeepsEqualKeysStable)
{
    const std::vector<double> x{1.0, 1.0, 0.0};
    const std::vector<double> y{1.0, 1.0, 0.0};
    const std::vector<double> z{1.0, 1.0, 0.0};

    const DeviceCoordinates coordinates(x, y, z);
    const MortonKeys morton(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    const MortonSnapshot actual = copyMortonToHost(morton);

    ASSERT_EQ(actual.keys.size(), 3u);
    ASSERT_EQ(actual.indices.size(), 3u);
    EXPECT_EQ(actual.indices, (std::vector<ParticleIndex>{2u, 0u, 1u}));
    EXPECT_EQ(actual.keys[1], actual.keys[2]);
}

TEST_F(MortonKeysTest, MapsOppositeCubeCornersToThirtyBitExtremes)
{
    const DeviceCoordinates coordinates(
        {0.0, 1.0},
        {0.0, 1.0},
        {0.0, 1.0});

    const MortonKeys morton(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    const MortonSnapshot actual = copyMortonToHost(morton);

    ASSERT_EQ(actual.keys.size(), 2u);
    EXPECT_EQ(actual.keys[0], 0u);
    EXPECT_EQ(actual.keys[1], 0x3FFFFFFFu);
    EXPECT_EQ(actual.indices, (std::vector<ParticleIndex>{0u, 1u}));
}

TEST_F(MortonKeysTest, HandlesCoincidentParticlesDeterministically)
{
    const DeviceCoordinates coordinates(
        {3.5, 3.5, 3.5, 3.5},
        {-4.0, -4.0, -4.0, -4.0},
        {8.25, 8.25, 8.25, 8.25});

    const MortonKeys morton(
        coordinates.x(),
        coordinates.y(),
        coordinates.z(),
        coordinates.size());

    const MortonSnapshot actual = copyMortonToHost(morton);

    ASSERT_EQ(actual.keys.size(), 4u);
    EXPECT_TRUE(std::all_of(
        actual.keys.begin(),
        actual.keys.end(),
        [&](MortonKey key)
        {
            return key == actual.keys.front();
        }));
    EXPECT_EQ(
        actual.indices,
        (std::vector<ParticleIndex>{0u, 1u, 2u, 3u}));
}

TEST_F(MortonKeysTest, RejectsEmptyOrNullInput)
{
    EXPECT_THROW(
        MortonKeys(nullptr, nullptr, nullptr, 0),
        std::invalid_argument);

    EXPECT_THROW(
        MortonKeys(nullptr, nullptr, nullptr, 1),
        std::invalid_argument);
}

TEST_F(MortonKeysTest, PropagatesNonFiniteCoordinateFailure)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const DeviceCoordinates coordinates(
        {0.0, nan},
        {0.0, 1.0},
        {0.0, 1.0});

    EXPECT_THROW(
        MortonKeys(
            coordinates.x(),
            coordinates.y(),
            coordinates.z(),
            coordinates.size()),
        std::domain_error);
}

} // namespace
