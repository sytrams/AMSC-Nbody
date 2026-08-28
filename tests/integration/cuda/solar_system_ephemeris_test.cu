#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_test.hpp"
#include "simulation.hpp"
#include "solar_system_reference_fixture.hpp"

namespace {

struct HostParticles {
  std::vector<double> mass;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  [[nodiscard]] std::size_t size() const noexcept { return mass.size(); }
};

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    std::random_device randomDevice;
    const auto token = (static_cast<std::uint64_t>(randomDevice()) << 32) |
                       static_cast<std::uint64_t>(randomDevice());
    path_ = std::filesystem::temp_directory_path() /
            ("nbody-solar-ephemeris-" + std::to_string(token));
    if (!std::filesystem::create_directory(path_))
      throw std::runtime_error("Could not create ephemeris test directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

void readBlock(std::ifstream &input, std::vector<double> &values,
               const char *name) {
  const auto bytes =
      static_cast<std::streamsize>(values.size() * sizeof(double));
  input.read(reinterpret_cast<char *>(values.data()), bytes);
  if (input.gcount() != bytes)
    throw std::runtime_error(std::string("Incomplete ") + name + " block");
}

HostParticles loadHostParticles(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open particle dataset: " +
                             path.string());

  std::uint64_t count = 0;
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!input || count == 0 ||
      count >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("Invalid particle count in " + path.string());
  }

  HostParticles particles;
  particles.mass.resize(static_cast<std::size_t>(count));
  particles.x.resize(static_cast<std::size_t>(count));
  particles.y.resize(static_cast<std::size_t>(count));
  particles.z.resize(static_cast<std::size_t>(count));
  particles.vx.resize(static_cast<std::size_t>(count));
  particles.vy.resize(static_cast<std::size_t>(count));
  particles.vz.resize(static_cast<std::size_t>(count));

  readBlock(input, particles.mass, "mass");
  readBlock(input, particles.x, "x position");
  readBlock(input, particles.y, "y position");
  readBlock(input, particles.z, "z position");
  readBlock(input, particles.vx, "x velocity");
  readBlock(input, particles.vy, "y velocity");
  readBlock(input, particles.vz, "z velocity");
  return particles;
}

void checkCuda(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

std::vector<double> copyDeviceValues(const double *deviceValues,
                                     std::size_t count) {
  std::vector<double> hostValues(count);
  checkCuda(cudaMemcpy(hostValues.data(), deviceValues, count * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "copy ephemeris simulation values");
  return hostValues;
}

HostParticles copyParticles(DeviceParticlesView view) {
  return {copyDeviceValues(view.mass, view.count),
          copyDeviceValues(view.x, view.count),
          copyDeviceValues(view.y, view.count),
          copyDeviceValues(view.z, view.count),
          copyDeviceValues(view.vx, view.count),
          copyDeviceValues(view.vy, view.count),
          copyDeviceValues(view.vz, view.count)};
}

double vectorError(double actualX, double actualY, double actualZ,
                   double expectedX, double expectedY, double expectedZ) {
  const double dx = actualX - expectedX;
  const double dy = actualY - expectedY;
  const double dz = actualZ - expectedZ;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

class SolarSystemEphemerisTest : public CudaTest {};

TEST_F(SolarSystemEphemerisTest, MatchesHorizonsReferenceAfterOneDay) {
  constexpr std::size_t steps = 96;
  constexpr double durationSeconds = 86'400.0;
  constexpr double timeStep = durationSeconds / static_cast<double>(steps);
  constexpr std::array<const char *, 10> bodyNames{
      "Sun",     "Mercury", "Venus",  "Earth",   "Mars",
      "Jupiter", "Saturn",  "Uranus", "Neptune", "Pluto"};

  // HORIZONS includes perturbations that this ten-point-mass fixture cannot
  // reproduce, especially from omitted moons. These per-body bounds retain a
  // strict integrator check while acknowledging that deliberate model gap.
  constexpr std::array<double, 10> positionToleranceMeters{
      5.0,       1'500.0,  150.0,    150'000.0, 50.0,
      115'000.0, 30'000.0, 15'000.0, 60'000.0,  1'200'000.0};
  constexpr std::array<double, 10> velocityToleranceMetersPerSecond{
      1.0e-3, 5.0e-3, 5.0e-3, 3.25, 1.0e-3, 2.1, 0.6, 0.3, 1.15, 27.0};

  const TemporaryDirectory temporaryDirectory;
  const SolarSystemReferenceDatasets datasets =
      writeSolarSystemReferenceDatasets(temporaryDirectory.path());

  const HostParticles initial = loadHostParticles(datasets.epochA);
  const HostParticles reference = loadHostParticles(datasets.epochB);
  ASSERT_EQ(initial.size(), bodyNames.size());
  ASSERT_EQ(reference.size(), bodyNames.size());
  for (std::size_t index = 0; index < bodyNames.size(); ++index)
    EXPECT_DOUBLE_EQ(initial.mass[index], reference.mass[index]);

  std::ifstream input(datasets.epochA, std::ios::binary);
  ASSERT_TRUE(input.is_open());
  Particles particles(input);
  Simulation simulation(std::move(particles), {timeStep, 0.0, 0.0});
  simulation.initialize();
  simulation.run(steps);

  EXPECT_EQ(simulation.stepNumber(), steps);
  EXPECT_DOUBLE_EQ(simulation.time(), durationSeconds);

  const HostParticles actual = copyParticles(simulation.particles());
  ASSERT_EQ(actual.size(), reference.size());
  for (std::size_t index = 0; index < reference.size(); ++index) {
    SCOPED_TRACE(std::string("body: ") + bodyNames[index]);
    EXPECT_DOUBLE_EQ(actual.mass[index], reference.mass[index]);

    const double positionError =
        vectorError(actual.x[index], actual.y[index], actual.z[index],
                    reference.x[index], reference.y[index], reference.z[index]);
    const double velocityError = vectorError(
        actual.vx[index], actual.vy[index], actual.vz[index],
        reference.vx[index], reference.vy[index], reference.vz[index]);
    EXPECT_LE(positionError, positionToleranceMeters[index])
        << "3D position error: " << positionError << " m";
    EXPECT_LE(velocityError, velocityToleranceMetersPerSecond[index])
        << "3D velocity error: " << velocityError << " m/s";
  }
}

} // namespace
