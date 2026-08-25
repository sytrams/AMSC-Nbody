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

    double* mass = nullptr;
    double* comX = nullptr;
    double* comY = nullptr;
    double* comZ = nullptr;

    int* visitCount = nullptr;
};
