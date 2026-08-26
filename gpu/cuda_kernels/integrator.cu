#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "integrator.hpp"

namespace
{
    constexpr int kThreadsPerBlock = 256;

    void checkCuda(cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }


    void validateTimeStep(double dt)
    {
        if (!std::isfinite(dt) || dt <= 0.0)
            throw std::invalid_argument("Leapfrog timestep must be finite and positive");
    }


    void validateAccelerationPointers(const double* accelerationX, const double* accelerationY, const double* accelerationZ)
    {
        if (accelerationX == nullptr || accelerationY == nullptr || accelerationZ == nullptr)
            throw std::invalid_argument("Leapfrog acceleration arrays must not be null");
    }


    void validateVelocityPointers(double* velocityX, double* velocityY, double* velocityZ)
    {
        if (velocityX == nullptr || velocityY == nullptr || velocityZ == nullptr)
            throw std::invalid_argument("Leapfrog velocity arrays must not be null");
    }


    void validatePositionPointers(double* positionX, double* positionY, double* positionZ)
    {
        if (positionX == nullptr || positionY == nullptr || positionZ == nullptr)
            throw std::invalid_argument("Leapfrog position arrays must not be null");
    }

} // namespace


__global__ void leapfrogKickDriftKernel(double* positionX, double* positionY, double* positionZ, double* velocityX, double* velocityY, double* velocityZ, const double* accelerationX, const double* accelerationY, const double* accelerationZ, std::size_t particleCount, double dt)
{
    const std::size_t particle = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (particle >= particleCount)
        return;

    const double halfDt = 0.5 * dt;

    //
    // ============================================================
    // First half kick
    // ============================================================
    //
    // v_(n+1/2) =
    //     v_n + (dt / 2) a_n
    //
    velocityX[particle] += halfDt * accelerationX[particle];
    velocityY[particle] += halfDt * accelerationY[particle];
    velocityZ[particle] += halfDt * accelerationZ[particle];

    //
    // ============================================================
    // Drift
    // ============================================================
    //
    // x_(n+1) =
    //     x_n + dt v_(n+1/2)
    //
    positionX[particle] += dt * velocityX[particle];
    positionY[particle] += dt * velocityY[particle];
    positionZ[particle] += dt * velocityZ[particle];
}


__global__ void leapfrogKickKernel(double* velocityX, double* velocityY, double* velocityZ, const double* accelerationX, const double* accelerationY, const double* accelerationZ, std::size_t particleCount, double dt)
{
    const std::size_t particle = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (particle >= particleCount)
        return;

    const double halfDt = 0.5 * dt;

    //
    // ============================================================
    // Second half kick
    // ============================================================
    //
    // v_(n+1) =
    //     v_(n+1/2) + (dt / 2) a_(n+1)
    //
    velocityX[particle] += halfDt * accelerationX[particle];
    velocityY[particle] += halfDt * accelerationY[particle];
    velocityZ[particle] += halfDt * accelerationZ[particle];
}


void leapfrogKickDrift(double* d_positionX, double* d_positionY, double* d_positionZ, double* d_velocityX, double* d_velocityY, double* d_velocityZ, const double* d_accelerationX, const double* d_accelerationY, const double* d_accelerationZ, std::size_t particleCount, double dt)
{
    validateTimeStep(dt);

    if (particleCount == 0)
        return;

    validatePositionPointers(d_positionX, d_positionY, d_positionZ);
    validateVelocityPointers(d_velocityX, d_velocityY, d_velocityZ);
    validateAccelerationPointers(d_accelerationX, d_accelerationY, d_accelerationZ);

    const std::size_t blocks = (particleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;

    leapfrogKickDriftKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock>>>(d_positionX, d_positionY, d_positionZ, d_velocityX, d_velocityY, d_velocityZ, d_accelerationX, d_accelerationY, d_accelerationZ, particleCount, dt);

    checkCuda(cudaGetLastError(), "leapfrogKickDriftKernel launch");
    checkCuda(cudaDeviceSynchronize(), "leapfrogKickDriftKernel execution");
}


void leapfrogKick(double* d_velocityX, double* d_velocityY, double* d_velocityZ, const double* d_accelerationX, const double* d_accelerationY, const double* d_accelerationZ, std::size_t particleCount, double dt)
{
    validateTimeStep(dt);

    if (particleCount == 0)
        return;

    validateVelocityPointers(d_velocityX, d_velocityY, d_velocityZ);
    validateAccelerationPointers(d_accelerationX, d_accelerationY, d_accelerationZ);

    const std::size_t blocks = (particleCount + kThreadsPerBlock - 1) / kThreadsPerBlock;

    leapfrogKickKernel<<<static_cast<unsigned int>(blocks), kThreadsPerBlock>>>(d_velocityX, d_velocityY, d_velocityZ, d_accelerationX, d_accelerationY, d_accelerationZ, particleCount, dt);

    checkCuda(cudaGetLastError(), "leapfrogKickKernel launch");
    checkCuda(cudaDeviceSynchronize(), "leapfrogKickKernel execution");
}