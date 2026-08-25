#pragma once

#include <cstdint>

#include "morton_leaf_groups.hpp"
#include "tree_types.hpp"

void allocateTree(Tree& tree, int nLeaves);

void buildTree(Tree& tree, const std::uint32_t* d_sortedKeys, int nLeaves);

void buildTreeFromMortonGroups(Tree& tree, const MortonLeafGroups& groups);

//mantiene validi i test precedenti e assume una foglia = una particella
void computeCenterOfMass(Tree& tree, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ, int nLeaves);

//nuova funzione da usare nella pipeline reale: una foglia = una chiave Morton = una o più particelle
void computeGroupedCenterOfMass(Tree& tree, const MortonLeafGroups& groups, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ);

void freeTree(Tree& tree);