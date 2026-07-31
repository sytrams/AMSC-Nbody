#pragma once    //impedisce inclusioni multiple

struct Tree
{
    int nBodies = 0;
    int nInternalNodes = 0;

    int* left = nullptr;
    int* right = nullptr;
    int* parent = nullptr;


    //physical data
    float* mass = nullptr;
    float* comX = nullptr;
    float* comY = nullptr;
    float* comZ = nullptr;

    float* size = nullptr;

    //bottom-up synchronization
    int* visitCount = nullptr;
};