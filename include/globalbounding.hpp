#pragma once
#include <cstddef>
#include <memory>


struct minmax{
    std::unique_ptr<double[]> min;
    std::unique_ptr<double[]> max;
};

void gpu_compute_boundaries(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, const std::unique_ptr<double[]>& z, const int N, minmax* results);
