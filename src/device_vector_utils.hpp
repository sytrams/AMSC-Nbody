#pragma once

#include <thrust/device_vector.h>
#include <thrust/memory.h>

namespace nbody {

template <typename T>
[[nodiscard]] T *deviceData(thrust::device_vector<T> &values) noexcept {
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
}

template <typename T>
[[nodiscard]] const T *
deviceData(const thrust::device_vector<T> &values) noexcept {
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
}

} // namespace nbody
