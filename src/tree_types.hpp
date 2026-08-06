#pragma once

struct Tree
{
    int nLeaves = 0;
    int nInternalNodes = 0;

    int* left = nullptr;
    int* right = nullptr;
    int* parent = nullptr;

    int* rangeFirst = nullptr;
    int* rangeLast = nullptr;

    int* prefixLength = nullptr;

    float* mass = nullptr;
    float* comX = nullptr;
    float* comY = nullptr;
    float* comZ = nullptr;

    int* visitCount = nullptr;
};
