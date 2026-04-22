module;

#include <math.h>
#include <array>
#include <vector>
#include <memory>

export module tree;

import particle;
import vector;
import system;

template <std::size_t DIM>
class Node{
    int* left, right, parent;
    float* mass, com, size;
};