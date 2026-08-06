#pragma once

#include <cstdint>

struct Octree
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

    float* mass = nullptr;
    float* comX = nullptr;
    float* comY = nullptr;
    float* comZ = nullptr;

    //Geometria della cella
    float* centerX = nullptr;
    float* centerY = nullptr;
    float* centerZ = nullptr;
    float* halfSie = nullptr;

    //Bottom-up synchronization
    int* pendingChildren = nullptr;
    int* childCount = nullptr;
};