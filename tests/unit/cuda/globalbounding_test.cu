#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_test.hpp"
#include "globalbounding.hpp"

namespace {

class DeviceCoordinates {
public:
  DeviceCoordinates(const std::vector<double> &x, const std::vector<double> &y,
                    const std::vector<double> &z)
      : x_(x.begin(), x.end()), y_(y.begin(), y.end()), z_(z.begin(), z.end()) {
    if (x.empty() || x.size() != y.size() || x.size() != z.size()) {
      throw std::invalid_argument("Test coordinates must have equal sizes.");
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return x_.size(); }

  [[nodiscard]] const double *x() const noexcept {
    return thrust::raw_pointer_cast(x_.data());
  }

  [[nodiscard]] const double *y() const noexcept {
    return thrust::raw_pointer_cast(y_.data());
  }

  [[nodiscard]] const double *z() const noexcept {
    return thrust::raw_pointer_cast(z_.data());
  }

private:
  thrust::device_vector<double> x_;
  thrust::device_vector<double> y_;
  thrust::device_vector<double> z_;
};

class GlobalBoundingTest : public CudaTest {};

TEST_F(GlobalBoundingTest, DefaultBoxHasInvalidSentinelSide) {
  const Bbox box;
  const auto values = box.values();

  EXPECT_DOUBLE_EQ(values.center_x, 0.0);
  EXPECT_DOUBLE_EQ(values.center_y, 0.0);
  EXPECT_DOUBLE_EQ(values.center_z, 0.0);
  EXPECT_DOUBLE_EQ(values.side, -1.0);
  EXPECT_NE(box.device_data(), nullptr);
}

TEST_F(GlobalBoundingTest, ComputesCenterAndLongestCubeSide) {
  const DeviceCoordinates coordinates({-2.0, 4.0, 1.0}, {10.0, 14.0, -2.0},
                                      {-1.0, 0.0, 5.0});

  const Bbox box(coordinates.x(), coordinates.y(), coordinates.z(),
                 coordinates.size());
  const auto values = box.values();

  EXPECT_DOUBLE_EQ(values.center_x, 1.0);
  EXPECT_DOUBLE_EQ(values.center_y, 6.0);
  EXPECT_DOUBLE_EQ(values.center_z, 2.0);
  EXPECT_DOUBLE_EQ(values.side, 16.0);
}

TEST_F(GlobalBoundingTest, ExposesCenterAndSideInDeviceMemory) {
  const DeviceCoordinates coordinates({-2.0, 4.0}, {10.0, 14.0}, {-1.0, 5.0});
  const Bbox box(coordinates.x(), coordinates.y(), coordinates.z(),
                 coordinates.size());
  std::array<double, Bbox::value_count> device_values{};

  const cudaError_t status =
      cudaMemcpy(device_values.data(), box.device_data(),
                 device_values.size() * sizeof(double), cudaMemcpyDeviceToHost);

  ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);
  EXPECT_DOUBLE_EQ(device_values[0], 1.0);
  EXPECT_DOUBLE_EQ(device_values[1], 12.0);
  EXPECT_DOUBLE_EQ(device_values[2], 2.0);
  EXPECT_DOUBLE_EQ(device_values[3], 6.0);
}

TEST_F(GlobalBoundingTest, UsesPositiveSideForCoincidentParticles) {
  const DeviceCoordinates coordinates({3.5, 3.5}, {-4.0, -4.0}, {8.25, 8.25});

  const Bbox box(coordinates.x(), coordinates.y(), coordinates.z(),
                 coordinates.size());
  const auto values = box.values();

  EXPECT_DOUBLE_EQ(values.center_x, 3.5);
  EXPECT_DOUBLE_EQ(values.center_y, -4.0);
  EXPECT_DOUBLE_EQ(values.center_z, 8.25);
  EXPECT_DOUBLE_EQ(values.side, 1.0);
}

TEST_F(GlobalBoundingTest, RecomputesAnExistingBox) {
  const DeviceCoordinates initial_coordinates({-1.0, 1.0}, {-1.0, 1.0},
                                              {-1.0, 1.0});
  Bbox box(initial_coordinates.x(), initial_coordinates.y(),
           initial_coordinates.z(), initial_coordinates.size());

  const DeviceCoordinates replacement_coordinates({10.0, 14.0}, {20.0, 22.0},
                                                  {30.0, 31.0});
  box.recompute(replacement_coordinates.x(), replacement_coordinates.y(),
                replacement_coordinates.z(), replacement_coordinates.size());
  const auto values = box.values();

  EXPECT_DOUBLE_EQ(values.center_x, 12.0);
  EXPECT_DOUBLE_EQ(values.center_y, 21.0);
  EXPECT_DOUBLE_EQ(values.center_z, 30.5);
  EXPECT_DOUBLE_EQ(values.side, 4.0);
}

TEST_F(GlobalBoundingTest, RejectsNanAndInfiniteCoordinates) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  {
    SCOPED_TRACE("NaN x coordinate");
    const DeviceCoordinates coordinates({0.0, nan}, {0.0, 1.0}, {0.0, 1.0});
    EXPECT_THROW(
        {
          Bbox box(coordinates.x(), coordinates.y(), coordinates.z(),
                   coordinates.size());
        },
        std::domain_error);
  }
  {
    SCOPED_TRACE("positive-infinity y coordinate");
    const DeviceCoordinates coordinates({0.0, 1.0}, {0.0, infinity},
                                        {0.0, 1.0});
    EXPECT_THROW(
        {
          Bbox box(coordinates.x(), coordinates.y(), coordinates.z(),
                   coordinates.size());
        },
        std::domain_error);
  }
  {
    SCOPED_TRACE("negative-infinity z coordinate");
    const DeviceCoordinates coordinates({0.0, 1.0}, {0.0, 1.0},
                                        {0.0, -infinity});
    EXPECT_THROW(
        {
          Bbox box(coordinates.x(), coordinates.y(), coordinates.z(),
                   coordinates.size());
        },
        std::domain_error);
  }
}

} // namespace
