#pragma once

#include <type_traits>

#include <thrust/device_vector.h>

#include "device_vector_utils.hpp"

// Trivially-copyable, non-owning kernel argument. Tree owns every allocation.
template <typename Int, typename Double> struct BasicTreeDeviceView {
  int nLeaves = 0;
  int nInternalNodes = 0;

  Int *left = nullptr;
  Int *right = nullptr;
  Int *parent = nullptr;

  Int *rangeFirst = nullptr;
  Int *rangeLast = nullptr;
  Int *prefixLength = nullptr;

  Double *mass = nullptr;
  Double *comX = nullptr;
  Double *comY = nullptr;
  Double *comZ = nullptr;

  Int *visitCount = nullptr;
};

using TreeDeviceView = BasicTreeDeviceView<int, double>;
using ConstTreeDeviceView = BasicTreeDeviceView<const int, const double>;
static_assert(std::is_trivially_copyable_v<TreeDeviceView>);
static_assert(std::is_trivially_copyable_v<ConstTreeDeviceView>);

struct Tree {
  int nLeaves = 0;
  int nInternalNodes = 0;

  thrust::device_vector<int> left;
  thrust::device_vector<int> right;
  thrust::device_vector<int> parent;

  thrust::device_vector<int> rangeFirst;
  thrust::device_vector<int> rangeLast;

  thrust::device_vector<int> prefixLength;

  thrust::device_vector<double> mass;
  thrust::device_vector<double> comX;
  thrust::device_vector<double> comY;
  thrust::device_vector<double> comZ;

  thrust::device_vector<int> visitCount;

  [[nodiscard]] TreeDeviceView device_view() noexcept {
    return {nLeaves,
            nInternalNodes,
            nbody::deviceData(left),
            nbody::deviceData(right),
            nbody::deviceData(parent),
            nbody::deviceData(rangeFirst),
            nbody::deviceData(rangeLast),
            nbody::deviceData(prefixLength),
            nbody::deviceData(mass),
            nbody::deviceData(comX),
            nbody::deviceData(comY),
            nbody::deviceData(comZ),
            nbody::deviceData(visitCount)};
  }

  [[nodiscard]] ConstTreeDeviceView device_view() const noexcept {
    return {nLeaves,
            nInternalNodes,
            nbody::deviceData(left),
            nbody::deviceData(right),
            nbody::deviceData(parent),
            nbody::deviceData(rangeFirst),
            nbody::deviceData(rangeLast),
            nbody::deviceData(prefixLength),
            nbody::deviceData(mass),
            nbody::deviceData(comX),
            nbody::deviceData(comY),
            nbody::deviceData(comZ),
            nbody::deviceData(visitCount)};
  }
};
