#pragma once

#include "device_buffer.hpp"

struct TreeView
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

struct ConstTreeView
{
    int nLeaves = 0;
    int nInternalNodes = 0;

    const int* left = nullptr;
    const int* right = nullptr;
    const int* parent = nullptr;

    const int* rangeFirst = nullptr;
    const int* rangeLast = nullptr;
    const int* prefixLength = nullptr;

    const double* mass = nullptr;
    const double* comX = nullptr;
    const double* comY = nullptr;
    const double* comZ = nullptr;

    const int* visitCount = nullptr;
};


struct Tree
{
    int nLeaves = 0;
    int nInternalNodes = 0;

    DeviceBuffer<int> left;
    DeviceBuffer<int> right;
    DeviceBuffer<int> parent;

    DeviceBuffer<int> rangeFirst;
    DeviceBuffer<int> rangeLast;
    DeviceBuffer<int> prefixLength;

    DeviceBuffer<double> mass;
    DeviceBuffer<double> comX;
    DeviceBuffer<double> comY;
    DeviceBuffer<double> comZ;

    DeviceBuffer<int> visitCount;


    [[nodiscard]]
    TreeView view() noexcept
    {
        return {
            nLeaves,
            nInternalNodes,
            left.data(),
            right.data(),
            parent.data(),
            rangeFirst.data(),
            rangeLast.data(),
            prefixLength.data(),
            mass.data(),
            comX.data(),
            comY.data(),
            comZ.data(),
            visitCount.data()
        };
    }


    [[nodiscard]]
    ConstTreeView view() const noexcept
    {
        return {
            nLeaves,
            nInternalNodes,
            left.data(),
            right.data(),
            parent.data(),
            rangeFirst.data(),
            rangeLast.data(),
            prefixLength.data(),
            mass.data(),
            comX.data(),
            comY.data(),
            comZ.data(),
            visitCount.data()
        };
    }
};