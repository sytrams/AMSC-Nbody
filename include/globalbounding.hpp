#pragma once

#include <cstddef>

#include <thrust/device_vector.h>

class Particles;

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
  explicit Bbox(Particles &bodies);

  void recompute(Particles &bodies);

  [[nodiscard]] BboxValues values() const;
  [[nodiscard]] const double *device_data() const noexcept;
  [[nodiscard]] double *device_data() noexcept;

private:
  thrust::device_vector<double> centre_and_side_;

  void compute_box(Particles &bodies);
};
