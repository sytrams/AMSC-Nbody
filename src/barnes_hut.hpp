#pragma once

#include <cstddef>
#include <cstdint>

#include "octree_types.hpp"

struct BarnesHutParameters
{
    double theta = 0.5;
    double softening = 1.0e-6;
    double gravitationalConstant = 6.67430e-11;
};

enum class BarnesHutParticleOrder
{
    original,
    morton
};

void computeBarnesHutAcceleration(const Octree& octree, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ, double* d_accelerationX, double* d_accelerationY, double* d_accelerationZ, std::size_t particleCount, const BarnesHutParameters& parameters, BarnesHutParticleOrder particleOrder = BarnesHutParticleOrder::original);