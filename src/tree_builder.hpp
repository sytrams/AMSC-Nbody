#pragma once

#include <cstdint>

#include "tree_types.hpp"

void allocateTree(Tree& tree, int N);

void buildTree(Tree& tree, const std::uint32_t* d_sortedKeys, int N);

void computeCenterOfMass(Tree& tree, const std::uint32_t* d_sortedIndices, const float* d_mass, const float* d_positionX, const float* d_positionY, const float* d_positionZ, int N);

void freeTree(Tree& tree);