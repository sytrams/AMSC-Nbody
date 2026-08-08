#pragma once

#include <cstdint>

#include "octree_types.hpp"

void allocateOctreePhysicalData(Octree& octree);

void computeOctreeMassAndCenterOfMass(Octree& octree, const std::uint32_t* d_sortedIndices, const float* d_mass, const float* d_positionX, const float* d_positionY, const float* d_positionZ);