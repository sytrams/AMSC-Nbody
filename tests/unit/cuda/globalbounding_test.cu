#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_test.hpp"
#include "globalbounding.hpp"
#include "particle.hpp"

namespace {

std::filesystem::path makeTemporaryParticleFile() {
  std::random_device random_device;
  const auto token = (static_cast<std::uint64_t>(random_device()) << 32) |
                     static_cast<std::uint64_t>(random_device());

  return std::filesystem::temp_directory_path() /
         ("nbody-bounds-" + std::to_string(token) + ".bin");
}

class GlobalBoundingTest : public CudaTest {
protected:
  const std::filesystem::path test_file = makeTemporaryParticleFile();

  std::unique_ptr<Particles> makeParticles(const std::vector<double> &x,
                                           const std::vector<double> &y,
                                           const std::vector<double> &z) {
    if (x.empty() || x.size() != y.size() || x.size() != z.size()) {
      throw std::invalid_argument("Test coordinates must have equal sizes.");
    }

    const std::vector<double> mass(x.size(), 1.0);
    const std::vector<double> velocity(x.size(), 0.0);
    const auto count = static_cast<std::uint64_t>(x.size());

    std::ofstream output(test_file, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      throw std::runtime_error("Failed to create particle test file.");
    }

    output.write(reinterpret_cast<const char *>(&count), sizeof(count));

    const auto writeBlock = [&output](const std::vector<double> &values) {
      output.write(
          reinterpret_cast<const char *>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(double)));
    };

    writeBlock(mass);
    writeBlock(x);
    writeBlock(y);
    writeBlock(z);
    writeBlock(velocity);
    writeBlock(velocity);
    writeBlock(velocity);
    output.close();

    if (!output) {
      throw std::runtime_error("Failed to write particle test file.");
    }

    std::ifstream input(test_file, std::ios::binary);
    return std::make_unique<Particles>(input);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(test_file, error);
  }
};

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
  auto particles =
      makeParticles({-2.0, 4.0, 1.0}, {10.0, 14.0, -2.0}, {-1.0, 0.0, 5.0});

  const Bbox box(*particles);
  const auto values = box.values();

  EXPECT_DOUBLE_EQ(values.center_x, 1.0);
  EXPECT_DOUBLE_EQ(values.center_y, 6.0);
  EXPECT_DOUBLE_EQ(values.center_z, 2.0);
  EXPECT_DOUBLE_EQ(values.side, 16.0);
}

TEST_F(GlobalBoundingTest, ExposesCenterAndSideInDeviceMemory) {
  auto particles = makeParticles({-2.0, 4.0}, {10.0, 14.0}, {-1.0, 5.0});
  const Bbox box(*particles);
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
  auto particles = makeParticles({3.5, 3.5}, {-4.0, -4.0}, {8.25, 8.25});

  const Bbox box(*particles);
  const auto values = box.values();

  EXPECT_DOUBLE_EQ(values.center_x, 3.5);
  EXPECT_DOUBLE_EQ(values.center_y, -4.0);
  EXPECT_DOUBLE_EQ(values.center_z, 8.25);
  EXPECT_DOUBLE_EQ(values.side, 1.0);
}

TEST_F(GlobalBoundingTest, RecomputesAnExistingBox) {
  auto initial_particles = makeParticles({-1.0, 1.0}, {-1.0, 1.0}, {-1.0, 1.0});
  Bbox box(*initial_particles);

  auto replacement_particles =
      makeParticles({10.0, 14.0}, {20.0, 22.0}, {30.0, 31.0});
  box.recompute(*replacement_particles);
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
    auto particles = makeParticles({0.0, nan}, {0.0, 1.0}, {0.0, 1.0});
    EXPECT_THROW({ Bbox box(*particles); }, std::domain_error);
  }
  {
    SCOPED_TRACE("positive-infinity y coordinate");
    auto particles = makeParticles({0.0, 1.0}, {0.0, infinity}, {0.0, 1.0});
    EXPECT_THROW({ Bbox box(*particles); }, std::domain_error);
  }
  {
    SCOPED_TRACE("negative-infinity z coordinate");
    auto particles = makeParticles({0.0, 1.0}, {0.0, 1.0}, {0.0, -infinity});
    EXPECT_THROW({ Bbox box(*particles); }, std::domain_error);
  }
}

} // namespace
