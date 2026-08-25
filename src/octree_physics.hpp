#pragma once

#include <cstdint>

#include "octree_types.hpp"

void allocateOctreePhysicalData(Octree& octree);

void computeOctreeMassAndCenterOfMass(Octree& octree, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ);