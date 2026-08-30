#pragma once

#include "morton_leaf_groups.hpp"
#include "octree_types.hpp"
#include "radix_to_octree.hpp"
#include "tree_types.hpp"

void allocateOctreeTopology(Octree& octree, int nNodes);

void buildSparseOctreeTopology(Octree& octree, const Tree& radixTree, const MortonLeafGroups& groups, const RadixToOctreePlan& plan);

void freeOctree(Octree& octree) noexcept;