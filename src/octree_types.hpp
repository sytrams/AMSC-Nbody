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