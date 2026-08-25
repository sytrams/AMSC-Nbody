#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

struct BboxValues {
  double center_x;
  double center_y;
  double center_z;
  double side;
};

class Bbox {
public:
  static constexpr std::size_t value_count = 4;

  Bbox();
  // Coordinate pointers must each reference at least count doubles in device
  // memory.
  Bbox(const double *d_x, const double *d_y, const double *d_z,
       std::size_t count);

  void recompute(const double *d_x, const double *d_y, const double *d_z,
                 std::size_t count);

  [[nodiscard]] BboxValues values() const;
  [[nodiscard]] const double *device_data() const noexcept;
  [[nodiscard]] double *device_data() noexcept;

private:
  thrust::device_vector<double> centre_and_side_;

  void compute_box(const double *d_x, const double *d_y, const double *d_z,
                   std::size_t count);
};
