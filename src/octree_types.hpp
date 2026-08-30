#pragma once

#include <cstdint>

#include "device_buffer.hpp"

struct OctreeView
{
    int nParticles = 0;
    int nLeaves = 0;
    int nNodes = 0;
    int root = -1;

    //Layout node-major: children[node * 8 + octant]
    int* children = nullptr;
    int* parent = nullptr;
    int* level = nullptr;
    std::uint32_t* prefix = nullptr;

    int* firstParticle = nullptr;
    int* particleCount = nullptr;

    double* mass = nullptr;
    double* comX = nullptr;
    double* comY = nullptr;
    double* comZ = nullptr;

    //Geometria della cella
    double* centerX = nullptr;
    double* centerY = nullptr;
    double* centerZ = nullptr;
    double* halfSize = nullptr;

    //Bottom-up synchronization
    int* pendingChildren = nullptr;
    int* childCount = nullptr;
};

struct ConstOctreeView
{
    int nParticles = 0;
    int nLeaves = 0;
    int nNodes = 0;
    int root = -1;

    const int* children = nullptr;

    const int* parent = nullptr;
    const int* level = nullptr;

    const std::uint32_t* prefix = nullptr;

    const int* firstParticle = nullptr;
    const int* particleCount = nullptr;

    const double* mass = nullptr;

    const double* comX = nullptr;
    const double* comY = nullptr;
    const double* comZ = nullptr;

    const double* centerX = nullptr;
    const double* centerY = nullptr;
    const double* centerZ = nullptr;

    const double* halfSize = nullptr;

    const int* pendingChildren = nullptr;
    const int* childCount = nullptr;
};


struct Octree
{
    int nParticles = 0;
    int nLeaves = 0;
    int nNodes = 0;
    int root = -1;

    DeviceBuffer<int> children;

    DeviceBuffer<int> parent;
    DeviceBuffer<int> level;

    DeviceBuffer<std::uint32_t> prefix;

    DeviceBuffer<int> firstParticle;
    DeviceBuffer<int> particleCount;

    DeviceBuffer<double> mass;

    DeviceBuffer<double> comX;
    DeviceBuffer<double> comY;
    DeviceBuffer<double> comZ;

    DeviceBuffer<double> centerX;
    DeviceBuffer<double> centerY;
    DeviceBuffer<double> centerZ;

    DeviceBuffer<double> halfSize;

    DeviceBuffer<int> pendingChildren;
    DeviceBuffer<int> childCount;


    [[nodiscard]]
    OctreeView view() noexcept
    {
        return {
            nParticles,
            nLeaves,
            nNodes,
            root,

            children.data(),

            parent.data(),
            level.data(),

            prefix.data(),

            firstParticle.data(),
            particleCount.data(),

            mass.data(),

            comX.data(),
            comY.data(),
            comZ.data(),

            centerX.data(),
            centerY.data(),
            centerZ.data(),

            halfSize.data(),

            pendingChildren.data(),
            childCount.data()
        };
    }


    [[nodiscard]]
    ConstOctreeView view() const noexcept
    {
        return {
            nParticles,
            nLeaves,
            nNodes,
            root,

            children.data(),

            parent.data(),
            level.data(),

            prefix.data(),

            firstParticle.data(),
            particleCount.data(),

            mass.data(),

            comX.data(),
            comY.data(),
            comZ.data(),

            centerX.data(),
            centerY.data(),
            centerZ.data(),

            halfSize.data(),

            pendingChildren.data(),
            childCount.data()
        };
    }
};