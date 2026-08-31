#include "simulation.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "barnes_hut.hpp"
#include "globalbounding.hpp"
#include "integrator.hpp"
#include "morton.hpp"
#include "octree_builder.hpp"
#include "octree_geometry.hpp"
#include "octree_physics.hpp"
#include "simulation_cuda.hpp"
#include "tree_builder.hpp"

namespace {

void validateConfig(const SimulationConfig &config) {
  if (!std::isfinite(config.timeStep) || config.timeStep <= 0.0)
    throw std::invalid_argument(
        "Simulation timeStep must be finite and positive");

  if (!std::isfinite(config.theta) || config.theta < 0.0)
    throw std::invalid_argument(
        "Simulation theta must be finite and non-negative");

  if (!std::isfinite(config.softening) || config.softening < 0.0)
    throw std::invalid_argument(
        "Simulation softening must be finite and non-negative");
}

int checkedParticleCount(std::size_t count) {
  if (count == 0)
    throw std::invalid_argument("Simulation requires at least one particle");

  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("Particle count exceeds the tree representation");

  return static_cast<int>(count);
}

void releaseSpatialResources(MortonLeafGroups &groups, Tree &radixTree,
                             RadixToOctreePlan &plan, Octree &octree) noexcept {
  freeOctree(octree);
  freeRadixToOctreePlan(plan);
  freeTree(radixTree);
  freeMortonLeafGroups(groups);
}

struct StagedSpatialResources {
  MortonLeafGroups groups{};
  Tree radixTree{};
  RadixToOctreePlan plan{};
  Octree octree{};

  StagedSpatialResources() = default;
  StagedSpatialResources(const StagedSpatialResources &) = delete;
  StagedSpatialResources &operator=(const StagedSpatialResources &) = delete;

  ~StagedSpatialResources() {
    releaseSpatialResources(groups, radixTree, plan, octree);
  }
};

} // namespace

Simulation::Simulation(Particles particles, SimulationConfig config)
    : config_(config), particles_(std::move(particles)) {
  validateConfig(config_);
  (void)checkedParticleCount(particles_.device_view().count);
}

Simulation::~Simulation() noexcept { releaseResources(); }

void Simulation::initialize() {
  if (state_ != State::created)
    throw std::logic_error("Simulation can only be initialized once");

  try {
    deviceState_ = nbody::detail::createSimulationDeviceState(
        particles_.device_view().count);
    rebuildSpatialStructure();

    // computeAccelerations writes the next buffers. Promote them to the
    // current acceleration before the first leapfrog step.
    computeAccelerations();
    nbody::detail::swapAccelerationBuffers(*deviceState_);

    state_ = State::ready;
  } catch (...) {
    releaseResources();
    state_ = State::failed;
    throw;
  }
}

void Simulation::step() {
  if (state_ != State::ready)
    throw std::logic_error("Simulation must be initialized before stepping");

  try {
    updatePositions();
    rebuildSpatialStructure();
    computeAccelerations();
    updateVelocities();
    nbody::detail::swapAccelerationBuffers(*deviceState_);

    time_ += config_.timeStep;
    ++stepNumber_;
  } catch (...) {
    // A CUDA operation may already have changed particle storage. Do not
    // permit another step from a partially updated state.
    state_ = State::failed;
    throw;
  }
}

void Simulation::run(std::size_t numberOfSteps) {
  if (state_ != State::ready)
    throw std::logic_error("Simulation must be initialized before running");

  for (std::size_t stepIndex = 0; stepIndex < numberOfSteps; ++stepIndex)
    step();
}

double Simulation::time() const noexcept { return time_; }

std::size_t Simulation::stepNumber() const noexcept { return stepNumber_; }

DeviceParticlesView Simulation::particles() noexcept {
  return particles_.device_view();
}

void Simulation::rebuildSpatialStructure() {
  const DeviceParticlesView particleView = particles_.device_view();
  const int particleCount = checkedParticleCount(particleView.count);

  const Bbox boundingBox(particleView.x, particleView.y, particleView.z, particleView.count);

  auto stagedMorton = std::make_unique<MortonKeys>(particleView.x, particleView.y, particleView.z, particleView.count, boundingBox);

  StagedSpatialResources staged;

  allocateMortonLeafGroups(staged.groups, particleCount);
  buildMortonLeafGroups(staged.groups, stagedMorton->keys_device_data(),
                        particleCount);

  allocateTreeTopology(staged.radixTree, staged.groups.nGroups);
  buildTreeFromMortonGroups(staged.radixTree, staged.groups);

  allocateRadixToOctreePlan(staged.plan, 2 * staged.groups.nGroups - 1);
  buildRadixToOctreePlan(staged.plan, staged.radixTree, staged.groups);

  allocateOctreeTopology(staged.octree, staged.plan.nOctreeNodes);
  buildSparseOctreeTopology(staged.octree, staged.radixTree, staged.groups,
                            staged.plan);
  allocateOctreeGeometry(staged.octree);
  computeOctreeGeometry(staged.octree, boundingBox);
  allocateOctreePhysicalData(staged.octree);
  computeOctreeMassAndCenterOfMass(
      staged.octree, stagedMorton->indices_device_data(), particleView.mass,
      particleView.x, particleView.y, particleView.z);

  // Commit only after every stage succeeds. The staged owner protects all
  // intermediate CUDA allocations if construction throws.
  releaseSpatialResources(groups_, radixTree_, plan_, octree_);

  groups_ = std::move(staged.groups);
  staged.groups = {};
  radixTree_ = std::move(staged.radixTree);
  staged.radixTree = {};
  plan_ = std::move(staged.plan);
  staged.plan = {};
  octree_ = std::move(staged.octree);
  staged.octree = {};
  mortonKeys_ = std::move(stagedMorton);
}

void Simulation::computeAccelerations() {
  if (deviceState_ == nullptr || mortonKeys_ == nullptr)
    throw std::logic_error("Simulation spatial/device state is not available");

  const DeviceParticlesView particleView = particles_.device_view();
  nbody::detail::gatherMortonOrderedParticleData(*deviceState_, mortonKeys_->indices_device_data(), particleView.mass, particleView.x, particleView.y, particleView.z, particleView.count);
  const nbody::detail::DeviceMortonParticleView mortonParticles =
    nbody::detail::mortonOrderedParticles(*deviceState_);
  const nbody::detail::DeviceAccelerationView output =
      nbody::detail::nextAcceleration(*deviceState_);
  computeBarnesHutAcceleration(
      octree_, mortonKeys_->indices_device_data(), mortonParticles.mass,
      mortonParticles.x, mortonParticles.y, mortonParticles.z, output.x, output.y,
      output.z, particleView.count,
      BarnesHutParameters{config_.theta, config_.softening, G},
      BarnesHutParticleOrder::morton);
}

void Simulation::updatePositions() {
  if (deviceState_ == nullptr)
    throw std::logic_error("Simulation device state is not available");

  const DeviceParticlesView particleView = particles_.device_view();
  const nbody::detail::DeviceAccelerationView acceleration =
      nbody::detail::currentAcceleration(*deviceState_);

  leapfrogKickDrift(particleView.x, particleView.y, particleView.z,
                    particleView.vx, particleView.vy, particleView.vz,
                    acceleration.x, acceleration.y, acceleration.z,
                    particleView.count, config_.timeStep);
}

void Simulation::updateVelocities() {
  if (deviceState_ == nullptr)
    throw std::logic_error("Simulation device state is not available");

  const DeviceParticlesView particleView = particles_.device_view();
  const nbody::detail::DeviceAccelerationView acceleration =
      nbody::detail::nextAcceleration(*deviceState_);

  leapfrogKick(particleView.vx, particleView.vy, particleView.vz,
               acceleration.x, acceleration.y, acceleration.z,
               particleView.count, config_.timeStep);
}

void Simulation::releaseResources() noexcept {
  mortonKeys_.reset();
  releaseSpatialResources(groups_, radixTree_, plan_, octree_);
  nbody::detail::destroySimulationDeviceState(deviceState_);
  deviceState_ = nullptr;
}
