#include "globalbounding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace {

// Intermediate state for the device reduction.  In addition to the extrema,
// carry an error flag so invalid coordinates cannot be hidden by fmin/fmax.
struct Bounds {
  double xmin, xmax;
  double ymin, ymax;
  double zmin, zmax;
  bool contains_non_finite;
};

struct ParticleToBounds {
  const double *x;
  const double *y;
  const double *z;

  __host__ __device__ Bounds operator()(std::size_t index) const {
    // Represent one particle as a zero-volume axis-aligned bounding box.
    const double x_value = x[index];
    const double y_value = y[index];
    const double z_value = z[index];
    const bool finite =
        ::isfinite(x_value) && ::isfinite(y_value) && ::isfinite(z_value);

    return {x_value, x_value, y_value, y_value, z_value, z_value, !finite};
  }
};

struct CombineBounds {
  __host__ __device__ Bounds operator()(const Bounds &a,
                                        const Bounds &b) const {
    // Component-wise min/max makes this operation associative, allowing
    // Thrust to combine partial results in any order on the GPU.
    return {::fmin(a.xmin, b.xmin),
            ::fmax(a.xmax, b.xmax),
            ::fmin(a.ymin, b.ymin),
            ::fmax(a.ymax, b.ymax),
            ::fmin(a.zmin, b.zmin),
            ::fmax(a.zmax, b.zmax),
            a.contains_non_finite || b.contains_non_finite};
  }
};

} // namespace

Bbox::Bbox() : centre_and_side_(value_count, 0.0) {
  // A negative side marks a default-constructed box as not yet computed.
  centre_and_side_[3] = -1.0;
}

Bbox::Bbox(const double *d_x, const double *d_y, const double *d_z,
           std::size_t count)
    : centre_and_side_(value_count, 0.0) {
  compute_box(d_x, d_y, d_z, count);
}

void Bbox::recompute(const double *d_x, const double *d_y, const double *d_z,
                     std::size_t count) {
  compute_box(d_x, d_y, d_z, count);
}

BboxValues Bbox::values() const {
  // The canonical values live on the device; copy the four-element layout to
  // the host before exposing it as a named aggregate.
  std::array<double, value_count> host_values{};
  thrust::copy(centre_and_side_.cbegin(), centre_and_side_.cend(),
               host_values.begin());

  return {host_values[0], host_values[1], host_values[2], host_values[3]};
}

const double *Bbox::device_data() const noexcept {
  return thrust::raw_pointer_cast(centre_and_side_.data());
}

double *Bbox::device_data() noexcept {
  return thrust::raw_pointer_cast(centre_and_side_.data());
}

void Bbox::compute_box(const double *d_x, const double *d_y, const double *d_z,
                       std::size_t count) {
  if (count == 0) {
    throw std::invalid_argument(
        "Cannot compute a bounding box for an empty particle set.");
  }

  const auto first = thrust::make_counting_iterator<std::size_t>(0);
  const auto last = first + count;
  const double infinity = std::numeric_limits<double>::infinity();

  // These extrema form the identity value for the component-wise reduction:
  // the first particle replaces all six infinities.
  const Bounds initial{infinity, -infinity, infinity, -infinity,
                       infinity, -infinity, false};

  // Convert each particle index to a degenerate box, then merge all boxes in
  // one device-side reduction.  Counting iterators avoid allocating an index
  // array solely to traverse the coordinate arrays.
  const Bounds result = thrust::transform_reduce(
      thrust::device, first, last, ParticleToBounds{d_x, d_y, d_z}, initial,
      CombineBounds{});

  if (result.contains_non_finite) {
    throw std::domain_error(
        "Particle positions must be finite to compute a bounding box.");
  }

  const double center_x = (result.xmin + result.xmax) * 0.5;
  const double center_y = (result.ymin + result.ymax) * 0.5;
  const double center_z = (result.zmin + result.zmax) * 0.5;

  // Use the longest axis so the resulting cube encloses all three axis-aligned
  // ranges while remaining centered on the original bounding box.
  double side = std::max({result.xmax - result.xmin, result.ymax - result.ymin,
                          result.zmax - result.zmin});

  // A positive side keeps later Morton normalization well-defined when every
  // particle occupies the same point.
  if (side == 0.0) {
    side = 1.0;
  }

  const std::array<double, value_count> host_values{center_x, center_y,
                                                    center_z, side};
  // Keep the compact [center_x, center_y, center_z, side] representation on the
  // device for kernels that normalize positions against this box.
  thrust::copy(host_values.begin(), host_values.end(),
               centre_and_side_.begin());
}
