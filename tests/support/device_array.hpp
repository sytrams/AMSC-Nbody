#pragma once

#include <vector>

#include <thrust/device_vector.h>

#include "device_vector_utils.hpp"

template <typename T> class DeviceArray {
public:
  explicit DeviceArray(const std::vector<T> &host)
      : values_(host.begin(), host.end()) {}

  [[nodiscard]] const T *get() const noexcept {
    return nbody::deviceData(values_);
  }

  [[nodiscard]] T *get() noexcept { return nbody::deviceData(values_); }

  [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

private:
  thrust::device_vector<T> values_;
};
