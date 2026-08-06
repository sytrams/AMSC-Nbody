#pragma once

#include <cstdint>

#include "morton_leaf_groups.hpp"
#include "tree_types.hpp"

void allocateTree(Tree& tree, int nLeaves);

void buildTree(Tree& tree, const std::uint32_t* d_sortedKeys, int nLeaves);

void buildTreeFromMortonGroups(Tree& tree, const MortonLeafGroups& groups);

//mantiene validi i test precedenti e assume una foglia = una particella
void computeCenterOfMass(Tree& tree, const std::uint32_t* d_sortedIndices, const float* d_mass, const float* d_positionX, const float* d_positionY, const float* d_positionZ, int nLeaves);

//nuova funzione da usare nella pipeline reale: una foglia = una chiave Morton = una o più particelle
void computeGroupedCenterOfMass(Tree& tree, const MortonLeafGroups& groups, const std::uint32_t* d_sortedIndices, const float* d_mass, const float* d_positionX, const float* d_positionY, const float* d_positionZ);

void freeTree(Tree& tree);