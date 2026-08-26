#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuda_test.hpp"
#include "simulation.hpp"

namespace {

struct ParticleData {
  std::vector<double> mass{2.0};
  std::vector<double> x{1.0};
  std::vector<double> y{2.0};
  std::vector<double> z{-1.0};
  std::vector<double> vx{2.0};
  std::vector<double> vy{-3.0};
  std::vector<double> vz{0.5};

  [[nodiscard]] std::size_t size() const noexcept { return mass.size(); }
};

class TemporaryParticleFile {
public:
  explicit TemporaryParticleFile(const ParticleData &particles)
      : path_(makePath()) {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output.is_open())
      throw std::runtime_error("Could not create simulation test input");

    const auto count = static_cast<std::uint64_t>(particles.size());
    output.write(reinterpret_cast<const char *>(&count), sizeof(count));
    writeBlock(output, particles.mass);
    writeBlock(output, particles.x);
    writeBlock(output, particles.y);
    writeBlock(output, particles.z);
    writeBlock(output, particles.vx);
    writeBlock(output, particles.vy);
    writeBlock(output, particles.vz);

    if (!output.good())
      throw std::runtime_error("Could not write simulation test input");
  }

  ~TemporaryParticleFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TemporaryParticleFile(const TemporaryParticleFile &) = delete;
  TemporaryParticleFile &operator=(const TemporaryParticleFile &) = delete;

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  static std::filesystem::path makePath() {
    std::random_device randomDevice;
    const auto token = (static_cast<std::uint64_t>(randomDevice()) << 32) |
                       static_cast<std::uint64_t>(randomDevice());
    return std::filesystem::temp_directory_path() /
           ("nbody-simulation-" + std::to_string(token) + ".bin");
  }

  static void writeBlock(std::ofstream &output,
                         const std::vector<double> &values) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(double)));
  }

  std::filesystem::path path_;
};

void checkCuda(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

std::unique_ptr<Simulation> loadSimulation(const std::filesystem::path &path,
                                           SimulationConfig config) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open simulation test input");

  Particles particles(input);
  return std::make_unique<Simulation>(std::move(particles), config);
}

std::vector<double> copyDeviceValues(const double *deviceValues,
                                     std::size_t count) {
  std::vector<double> hostValues(count);
  checkCuda(cudaMemcpy(hostValues.data(), deviceValues, count * sizeof(double),
                       cudaMemcpyDeviceToHost),
            "copy simulation values");
  return hostValues;
}

class SimulationTest : public CudaTest {};

TEST_F(SimulationTest, RejectsInvalidConfiguration) {
  const TemporaryParticleFile file{ParticleData{}};
  const double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(loadSimulation(file.path(), {0.0, 0.5, 0.01}),
               std::invalid_argument);
  EXPECT_THROW(loadSimulation(file.path(), {-1.0, 0.5, 0.01}),
               std::invalid_argument);
  EXPECT_THROW(loadSimulation(file.path(), {nan, 0.5, 0.01}),
               std::invalid_argument);
  EXPECT_THROW(loadSimulation(file.path(), {1.0, -0.1, 0.01}),
               std::invalid_argument);
  EXPECT_THROW(loadSimulation(file.path(), {1.0, nan, 0.01}),
               std::invalid_argument);
  EXPECT_THROW(loadSimulation(file.path(), {1.0, 0.5, -0.01}),
               std::invalid_argument);
  EXPECT_THROW(loadSimulation(file.path(), {1.0, 0.5, nan}),
               std::invalid_argument);
}

TEST_F(SimulationTest, RequiresInitializationBeforeAdvancing) {
  const TemporaryParticleFile file{ParticleData{}};
  auto simulation = loadSimulation(file.path(), {0.25, 0.5, 0.01});

  EXPECT_THROW(simulation->step(), std::logic_error);
  EXPECT_THROW(simulation->run(1), std::logic_error);
}

TEST_F(SimulationTest, InitializesOnlyOnceWithoutAdvancingState) {
  const ParticleData initial;
  const TemporaryParticleFile file{initial};
  auto simulation = loadSimulation(file.path(), {0.25, 0.7, 0.02});

  simulation->initialize();

  const DeviceParticlesView view = simulation->particles();
  const auto actualX = copyDeviceValues(view.x, view.count);
  const auto actualVx = copyDeviceValues(view.vx, view.count);

  ASSERT_EQ(view.count, initial.size());
  EXPECT_DOUBLE_EQ(simulation->time(), 0.0);
  EXPECT_EQ(simulation->stepNumber(), 0u);
  EXPECT_DOUBLE_EQ(actualX.front(), initial.x.front());
  EXPECT_DOUBLE_EQ(actualVx.front(), initial.vx.front());
  EXPECT_THROW(simulation->initialize(), std::logic_error);
}

TEST_F(SimulationTest, AdvancesAnIsolatedParticleAtConstantVelocity) {
  const ParticleData initial;
  const TemporaryParticleFile file{initial};
  constexpr double timeStep = 0.25;
  constexpr std::size_t steps = 4;
  auto simulation = loadSimulation(file.path(), {timeStep, 0.6, 0.03});

  simulation->initialize();
  simulation->run(steps);

  const double elapsed = timeStep * static_cast<double>(steps);
  const DeviceParticlesView view = simulation->particles();
  const auto actualX = copyDeviceValues(view.x, view.count);
  const auto actualY = copyDeviceValues(view.y, view.count);
  const auto actualZ = copyDeviceValues(view.z, view.count);
  const auto actualVx = copyDeviceValues(view.vx, view.count);
  const auto actualVy = copyDeviceValues(view.vy, view.count);
  const auto actualVz = copyDeviceValues(view.vz, view.count);

  ASSERT_EQ(view.count, initial.size());
  EXPECT_DOUBLE_EQ(simulation->time(), elapsed);
  EXPECT_EQ(simulation->stepNumber(), steps);
  EXPECT_NEAR(actualX.front(), initial.x.front() + initial.vx.front() * elapsed,
              1.0e-12);
  EXPECT_NEAR(actualY.front(), initial.y.front() + initial.vy.front() * elapsed,
              1.0e-12);
  EXPECT_NEAR(actualZ.front(), initial.z.front() + initial.vz.front() * elapsed,
              1.0e-12);
  EXPECT_DOUBLE_EQ(actualVx.front(), initial.vx.front());
  EXPECT_DOUBLE_EQ(actualVy.front(), initial.vy.front());
  EXPECT_DOUBLE_EQ(actualVz.front(), initial.vz.front());
}

} // namespace
