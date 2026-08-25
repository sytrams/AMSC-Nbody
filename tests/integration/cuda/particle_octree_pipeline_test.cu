#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cuda_test.hpp"
#include "morton.hpp"
#include "morton_leaf_groups.hpp"
#include "octree_builder.hpp"
#include "octree_physics.hpp"
#include "particle.hpp"
#include "radix_to_octree.hpp"
#include "tree_builder.hpp"

namespace {

struct ParticleData {
  std::vector<double> mass{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> x{-3.0, 4.0, 0.0, 0.0, 2.0, -1.0};
  std::vector<double> y{-1.0, 2.0, 5.0, 5.0, -4.0, 0.0};
  std::vector<double> z{2.0, -2.0, 1.0, 1.0, 3.0, -5.0};
  std::vector<double> vx{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> vy{-0.1, -0.2, -0.3, -0.4, -0.5, -0.6};
  std::vector<double> vz{0.0, 0.1, 0.0, -0.1, 0.2, -0.2};

  [[nodiscard]] std::size_t size() const noexcept { return mass.size(); }
};

class TemporaryParticleFile {
public:
  explicit TemporaryParticleFile(const ParticleData &particles)
      : path_(makePath()) {
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      throw std::runtime_error("Could not create temporary particle file");
    }

    const auto count = static_cast<std::uint64_t>(particles.size());
    output.write(reinterpret_cast<const char *>(&count), sizeof(count));
    writeBlock(output, particles.mass);
    writeBlock(output, particles.x);
    writeBlock(output, particles.y);
    writeBlock(output, particles.z);
    writeBlock(output, particles.vx);
    writeBlock(output, particles.vy);
    writeBlock(output, particles.vz);

    if (!output.good()) {
      throw std::runtime_error("Could not write temporary particle file");
    }
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
           ("nbody-pipeline-" + std::to_string(token) + ".bin");
  }

  static void writeBlock(std::ofstream &output,
                         const std::vector<double> &values) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(double)));
  }

  std::filesystem::path path_;
};

struct PipelineResources {
  MortonLeafGroups groups{};
  Tree radixTree{};
  RadixToOctreePlan plan{};
  Octree octree{};

  ~PipelineResources() {
    freeOctree(octree);
    freeRadixToOctreePlan(plan);
    freeTree(radixTree);
    freeMortonLeafGroups(groups);
  }
};

struct ConvertToFloat {
  __host__ __device__ float operator()(double value) const {
    return static_cast<float>(value);
  }
};

thrust::device_vector<float> convertToFloat(const double *input,
                                            std::size_t count) {
  thrust::device_vector<float> output(count);
  thrust::transform(thrust::device_pointer_cast(input),
                    thrust::device_pointer_cast(input) + count, output.begin(),
                    ConvertToFloat{});
  return output;
}

std::array<float, 4> copyRootPhysics(const Octree &octree) {
  std::array<float, 4> root{};
  const std::array<const float *, 4> fields{octree.mass, octree.comX,
                                            octree.comY, octree.comZ};

  for (std::size_t field = 0; field < fields.size(); ++field) {
    const cudaError_t status = cudaMemcpy(
        &root[field], fields[field], sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("copy octree root: ") +
                               cudaGetErrorString(status));
    }
  }
  return root;
}

std::array<double, 4> referenceRoot(const ParticleData &particles) {
  double totalMass = 0.0;
  double weightedX = 0.0;
  double weightedY = 0.0;
  double weightedZ = 0.0;

  for (std::size_t particle = 0; particle < particles.size(); ++particle) {
    const double mass = particles.mass[particle];
    totalMass += mass;
    weightedX += mass * particles.x[particle];
    weightedY += mass * particles.y[particle];
    weightedZ += mass * particles.z[particle];
  }

  return {totalMass, weightedX / totalMass, weightedY / totalMass,
          weightedZ / totalMass};
}

class ParticleOctreePipelineTest : public CudaTest {};

TEST_F(ParticleOctreePipelineTest, ComputesRootFromLoadedParticles) {
  const ParticleData expectedParticles;
  const TemporaryParticleFile file(expectedParticles);
  std::ifstream input(file.path(), std::ios::binary);
  ASSERT_TRUE(input.is_open());

  Particles particles(input);
  const DeviceParticlesView particleView = particles.device_view();
  ASSERT_EQ(particleView.count, expectedParticles.size());

  const MortonKeys morton(particleView.x, particleView.y, particleView.z,
                          particleView.count);
  PipelineResources resources;
  const int particleCount = static_cast<int>(particleView.count);

  allocateMortonLeafGroups(resources.groups, particleCount);
  buildMortonLeafGroups(resources.groups, morton.keys_device_data(),
                        particleCount);
  ASSERT_GT(resources.groups.nGroups, 0);
  EXPECT_LT(resources.groups.nGroups, particleCount);

  allocateTree(resources.radixTree, resources.groups.nGroups);
  buildTreeFromMortonGroups(resources.radixTree, resources.groups);

  allocateRadixToOctreePlan(resources.plan, 2 * resources.groups.nGroups - 1);
  buildRadixToOctreePlan(resources.plan, resources.radixTree, resources.groups);

  allocateOctreeTopology(resources.octree, resources.plan.nOctreeNodes);
  buildSparseOctreeTopology(resources.octree, resources.radixTree,
                            resources.groups, resources.plan);
  allocateOctreePhysicalData(resources.octree);

  auto mass = convertToFloat(particleView.mass, particleView.count);
  auto x = convertToFloat(particleView.x, particleView.count);
  auto y = convertToFloat(particleView.y, particleView.count);
  auto z = convertToFloat(particleView.z, particleView.count);

  computeOctreeMassAndCenterOfMass(
      resources.octree, morton.indices_device_data(),
      thrust::raw_pointer_cast(mass.data()), thrust::raw_pointer_cast(x.data()),
      thrust::raw_pointer_cast(y.data()), thrust::raw_pointer_cast(z.data()));

  const auto actualRoot = copyRootPhysics(resources.octree);
  const auto expectedRoot = referenceRoot(expectedParticles);

  EXPECT_NEAR(actualRoot[0], expectedRoot[0], 1.0e-5);
  EXPECT_NEAR(actualRoot[1], expectedRoot[1], 1.0e-5);
  EXPECT_NEAR(actualRoot[2], expectedRoot[2], 1.0e-5);
  EXPECT_NEAR(actualRoot[3], expectedRoot[3], 1.0e-5);
}

} // namespace
