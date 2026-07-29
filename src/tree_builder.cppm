module;

#include <cstdint>

export module tree.builder;
import tree.types;

export void allocateTree(Tree& tree, int N);

export void buildTree(Tree& tree, const std::uint32_t* d_keys, int N);

export void freeTree(Tree& tree);