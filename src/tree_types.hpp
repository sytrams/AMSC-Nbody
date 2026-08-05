#pragma once

struct Tree
{
    int nBodies = 0;
    int nInternalNodes = 0;

    // Binary-radix-tree topology.
    // Leaves use global node IDs [0, nBodies).
    // Internal nodes use global node IDs [nBodies, 2 * nBodies - 1).
    // left/right and the metadata below are indexed by local internal-node ID
    // [0, nInternalNodes).
    int* left = nullptr;
    int* right = nullptr;
    int* parent = nullptr;

    // Sorted Morton-key range covered by each internal node.
    int* rangeFirst = nullptr;
    int* rangeLast = nullptr;

    // Longest common prefix of the first and last key in the node range.
    // With duplicate Morton codes this can exceed 32 because the sorted index
    // is used as a virtual suffix, exactly as in the Karras construction.
    int* prefixLength = nullptr;

    // Physical data, indexed by global node ID.
    float* mass = nullptr;
    float* comX = nullptr;
    float* comY = nullptr;
    float* comZ = nullptr;

    // Bottom-up synchronization, indexed by local internal-node ID.
    int* visitCount = nullptr;
};
