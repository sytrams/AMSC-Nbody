#pragma once

#include <cstdint>
#include <type_traits>

#include <thrust/device_vector.h>

#include "device_vector_utils.hpp"

// Trivially-copyable, non-owning kernel argument. Octree owns every allocation.
template <typename Int, typename Uint, typename Double>
struct BasicOctreeDeviceView {
  int nParticles = 0;
  int nLeaves = 0;
  int nNodes = 0;
  int root = -1;

  Int *children = nullptr;
  Int *parent = nullptr;
  Int *level = nullptr;
  Uint *prefix = nullptr;
  Int *firstParticle = nullptr;
  Int *particleCount = nullptr;
  Double *mass = nullptr;
  Double *comX = nullptr;
  Double *comY = nullptr;
  Double *comZ = nullptr;
  Double *centerX = nullptr;
  Double *centerY = nullptr;
  Double *centerZ = nullptr;
  Double *halfSize = nullptr;
  Int *pendingChildren = nullptr;
  Int *childCount = nullptr;
};

using OctreeDeviceView = BasicOctreeDeviceView<int, std::uint32_t, double>;
using ConstOctreeDeviceView =
    BasicOctreeDeviceView<const int, const std::uint32_t, const double>;
static_assert(std::is_trivially_copyable_v<OctreeDeviceView>);
static_assert(std::is_trivially_copyable_v<ConstOctreeDeviceView>);

struct Octree {
  int nParticles = 0;
  int nLeaves = 0;
  int nNodes = 0;
  int root = -1;

  // Layout node-major: children[node * 8 + octant]
  thrust::device_vector<int> children;
  thrust::device_vector<int> parent;
  thrust::device_vector<int> level;
  thrust::device_vector<std::uint32_t> prefix;

  thrust::device_vector<int> firstParticle;
  thrust::device_vector<int> particleCount;

  thrust::device_vector<double> mass;
  thrust::device_vector<double> comX;
  thrust::device_vector<double> comY;
  thrust::device_vector<double> comZ;

  // Geometria della cella
  thrust::device_vector<double> centerX;
  thrust::device_vector<double> centerY;
  thrust::device_vector<double> centerZ;
  thrust::device_vector<double> halfSize;

  // Bottom-up synchronization
  thrust::device_vector<int> pendingChildren;
  thrust::device_vector<int> childCount;

  [[nodiscard]] OctreeDeviceView device_view() noexcept {
    return {nParticles,
            nLeaves,
            nNodes,
            root,
            nbody::deviceData(children),
            nbody::deviceData(parent),
            nbody::deviceData(level),
            nbody::deviceData(prefix),
            nbody::deviceData(firstParticle),
            nbody::deviceData(particleCount),
            nbody::deviceData(mass),
            nbody::deviceData(comX),
            nbody::deviceData(comY),
            nbody::deviceData(comZ),
            nbody::deviceData(centerX),
            nbody::deviceData(centerY),
            nbody::deviceData(centerZ),
            nbody::deviceData(halfSize),
            nbody::deviceData(pendingChildren),
            nbody::deviceData(childCount)};
  }

  [[nodiscard]] ConstOctreeDeviceView device_view() const noexcept {
    return {nParticles,
            nLeaves,
            nNodes,
            root,
            nbody::deviceData(children),
            nbody::deviceData(parent),
            nbody::deviceData(level),
            nbody::deviceData(prefix),
            nbody::deviceData(firstParticle),
            nbody::deviceData(particleCount),
            nbody::deviceData(mass),
            nbody::deviceData(comX),
            nbody::deviceData(comY),
            nbody::deviceData(comZ),
            nbody::deviceData(centerX),
            nbody::deviceData(centerY),
            nbody::deviceData(centerZ),
            nbody::deviceData(halfSize),
            nbody::deviceData(pendingChildren),
            nbody::deviceData(childCount)};
  }
};
