#include "globalbounding.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <cmath>

struct BBox {
    double min_x, max_x;
    double min_y, max_y;
    double min_z, max_z;

    __host__ __device__
    BBox() : min_x(INFINITY), max_x(-INFINITY),
             min_y(INFINITY), max_y(-INFINITY),
             min_z(INFINITY), max_z(-INFINITY) {}

    __host__ __device__
    BBox(double x, double y, double z) : min_x(x), max_x(x),
                                         min_y(y), max_y(y),
                                         min_z(z), max_z(z) {}
};

struct BBoxReduction {
    __host__ __device__
    BBox operator()(const BBox& a, const BBox& b) const {
        BBox res;
        res.min_x = fmin(a.min_x, b.min_x);
        res.max_x = fmax(a.max_x, b.max_x);
        res.min_y = fmin(a.min_y, b.min_y);
        res.max_y = fmax(a.max_y, b.max_y);
        res.min_z = fmin(a.min_z, b.min_z);
        res.max_z = fmax(a.max_z, b.max_z);
        return res;
    }
};

struct TupleToBBox {
    __host__ __device__
    BBox operator()(const thrust::tuple<double, double, double>& t) const {
        return BBox(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t));
    }
};

void gpu_compute_boundaries(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, const std::unique_ptr<double[]>& z, const int N, minmax* results){
    if (N <= 0) return;

    // Allocate device memory and copy data
    thrust::device_vector<double> d_x(x.get(), x.get() + N);
    thrust::device_vector<double> d_y(y.get(), y.get() + N);
    thrust::device_vector<double> d_z(z.get(), z.get() + N);

    // Create zip iterator to process x, y, z together
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin(), d_z.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_x.end(), d_y.end(), d_z.end()));

    // Perform reduction
    BBox result = thrust::transform_reduce(
        thrust::device,
        begin,
        end,
        TupleToBBox(),
        BBox(),
        BBoxReduction()
    );

    // Allocate results if needed
    if (!results->min) results->min = std::make_unique<double[]>(3);
    if (!results->max) results->max = std::make_unique<double[]>(3);

    // Store back to host
    results->min[0] = result.min_x;
    results->min[1] = result.min_y;
    results->min[2] = result.min_z;

    results->max[0] = result.max_x;
    results->max[1] = result.max_y;
    results->max[2] = result.max_z;
}
