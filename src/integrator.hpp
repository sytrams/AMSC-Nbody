#pragma once

#include <cstddef>

void leapfrogKickDrift(double* d_positionX, double* d_positionY, double* d_positionZ, double* d_velocityX, double* d_velocityY, double* d_velocityZ, const double* d_accelerationX, const double* d_accelerationY, const double* d_accelerationZ, std::size_t particleCount, double dt);

void leapfrogKick(double* d_velocityX, double* d_velocityY, double* d_velocityZ, const double* d_accelerationX, const double* d_accelerationY, const double* d_accelerationZ, std::size_t particleCount, double dt); 