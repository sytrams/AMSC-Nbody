#pragma once

#include <cstdint>

#include "tree_types.hpp"

void allocateTree(Tree& tree, int N);

void buildTree(Tree& tree, const std::uint32_t* d_keys, int N);

void freeTree(Tree& tree);