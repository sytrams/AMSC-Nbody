#pragma once

#include <cstddef>
#include <memory>

#include "morton_leaf_groups.hpp"
#include "octree_types.hpp"
#include "particle.hpp"
#include "radix_to_octree.hpp"
#include "tree_types.hpp"

class MortonKeys;

namespace nbody::detail {
struct SimulationDeviceState;
}

struct SimulationConfig {
  double timeStep;
  double theta;
  double softening;
  bool profileStages = false;
};

class Simulation {
public:
  Simulation(Particles particles, SimulationConfig config);
  ~Simulation() noexcept;

  Simulation(const Simulation &) = delete;
  Simulation &operator=(const Simulation &) = delete;

  void initialize();
  void step();
  void run(std::size_t numberOfSteps);

  [[nodiscard]] double time() const noexcept;
  [[nodiscard]] std::size_t stepNumber() const noexcept;
  [[nodiscard]] DeviceParticlesView particles() noexcept;

private:
  enum class State { created, ready, failed };

  void rebuildSpatialStructure();
  void computeAccelerations();
  void updatePositions();
  void updateVelocities();
  void releaseResources() noexcept;

  SimulationConfig config_;
  Particles particles_;

  MortonLeafGroups groups_{};
  Tree radixTree_{};
  RadixToOctreePlan plan_{};
  Octree octree_{};

  // Morton indices must remain alive while the octree refers to their
  // sorted-particle order. Acceleration storage is kept opaque so CUDA-only
  // details do not leak into this public header.
  std::unique_ptr<MortonKeys> mortonKeys_;
  nbody::detail::SimulationDeviceState *deviceState_ = nullptr;

  double time_ = 0.0;
  std::size_t stepNumber_ = 0;
  State state_ = State::created;
};
