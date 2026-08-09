#pragma once

#include <cstdint>
#include <fstream>
#include <memory>

inline constexpr double G = 6.67430e-11;
inline constexpr double K = 8.98755e9;

class Particles
{
public:
    explicit Particles(std::ifstream& input);

    std::uint32_t num_particles = 0;
    std::unique_ptr<double[]> mass;
    std::unique_ptr<double[]> x;
    std::unique_ptr<double[]> y;
    std::unique_ptr<double[]> z;
    std::unique_ptr<double[]> vx;
    std::unique_ptr<double[]> vy;
    std::unique_ptr<double[]> vz;
};
