#pragma once

#include "globalbounding.hpp"
#include "octree_types.hpp"

// Allocate the geometric data associated with every octree node.
void allocateOctreeGeometry(Octree& octree);

// Compute the geometric cube represented by every octree node.
void computeOctreeGeometry(Octree& octree, const Bbox& boundingBox);