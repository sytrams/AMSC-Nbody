#include "globalbounding.hpp"
#include "particle.hpp"

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
  centre_and_side_[3] = -1.0;
}

Bbox::Bbox(Particles &bodies) : centre_and_side_(value_count, 0.0) {
  compute_box(bodies);
}

void Bbox::recompute(Particles &bodies) { compute_box(bodies); }

BboxValues Bbox::values() const {
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

void Bbox::compute_box(Particles &bodies) {
  const auto particles = bodies.device_view();
  const auto count = particles.count;

  if (count == 0) {
    throw std::invalid_argument(
        "Cannot compute a bounding box for an empty particle set.");
  }

  const auto first = thrust::make_counting_iterator<std::size_t>(0);
  const auto last = first + count;
  const double infinity = std::numeric_limits<double>::infinity();

  const Bounds initial{infinity, -infinity, infinity, -infinity,
                       infinity, -infinity, false};

  const Bounds result = thrust::transform_reduce(
      thrust::device, first, last,
      ParticleToBounds{particles.x, particles.y, particles.z}, initial,
      CombineBounds{});

  if (result.contains_non_finite) {
    throw std::domain_error(
        "Particle positions must be finite to compute a bounding box.");
  }

  const double center_x = (result.xmin + result.xmax) * 0.5;
  const double center_y = (result.ymin + result.ymax) * 0.5;
  const double center_z = (result.zmin + result.zmax) * 0.5;
  double side = std::max({result.xmax - result.xmin, result.ymax - result.ymin,
                          result.zmax - result.zmin});

  // A positive side keeps later Morton normalization well-defined when every
  // particle occupies the same point.
  if (side == 0.0) {
    side = 1.0;
  }

  const std::array<double, value_count> host_values{center_x, center_y,
                                                    center_z, side};
  thrust::copy(host_values.begin(), host_values.end(),
               centre_and_side_.begin());
}
