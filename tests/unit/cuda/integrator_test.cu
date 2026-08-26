#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "cuda_test.hpp"
#include "integrator.hpp"

namespace
{

template <typename T>
std::vector<T> copyToHost(
    const thrust::device_vector<T>& device)
{
    std::vector<T> host(device.size());

    thrust::copy(
        device.begin(),
        device.end(),
        host.begin());

    return host;
}


class IntegratorTest : public CudaTest
{
};


TEST_F(
    IntegratorTest,
    KickDriftUpdatesVelocityAndPosition)
{
    //
    // Initial state:
    //
    // x  = 10
    // vx = 2
    // ax = 4
    //
    // dt = 0.5
    //
    // Half kick:
    //
    // vx_half =
    //     2 + 0.5 * 0.5 * 4
    //     = 3
    //
    // Drift:
    //
    // x_new =
    //     10 + 0.5 * 3
    //     = 11.5
    //

    thrust::device_vector<double> x{10.0};
    thrust::device_vector<double> y{20.0};
    thrust::device_vector<double> z{30.0};

    thrust::device_vector<double> vx{2.0};
    thrust::device_vector<double> vy{4.0};
    thrust::device_vector<double> vz{6.0};

    thrust::device_vector<double> ax{4.0};
    thrust::device_vector<double> ay{-2.0};
    thrust::device_vector<double> az{8.0};

    const double dt = 0.5;

    leapfrogKickDrift(
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(vx.data()),
        thrust::raw_pointer_cast(vy.data()),
        thrust::raw_pointer_cast(vz.data()),
        thrust::raw_pointer_cast(ax.data()),
        thrust::raw_pointer_cast(ay.data()),
        thrust::raw_pointer_cast(az.data()),
        1,
        dt);

    const auto hostX = copyToHost(x);
    const auto hostY = copyToHost(y);
    const auto hostZ = copyToHost(z);

    const auto hostVx = copyToHost(vx);
    const auto hostVy = copyToHost(vy);
    const auto hostVz = copyToHost(vz);

    EXPECT_DOUBLE_EQ(
        hostVx[0],
        3.0);

    EXPECT_DOUBLE_EQ(
        hostVy[0],
        3.5);

    EXPECT_DOUBLE_EQ(
        hostVz[0],
        8.0);

    EXPECT_DOUBLE_EQ(
        hostX[0],
        11.5);

    EXPECT_DOUBLE_EQ(
        hostY[0],
        21.75);

    EXPECT_DOUBLE_EQ(
        hostZ[0],
        34.0);
}


TEST_F(
    IntegratorTest,
    SecondKickCompletesVelocityUpdate)
{
    //
    // Assume that kick + drift already produced:
    //
    // vx_half = 3
    //
    // and the newly computed acceleration is:
    //
    // ax_new = 6
    //
    // dt = 0.5
    //
    // Then:
    //
    // vx_new =
    //     3 + 0.5 * 0.5 * 6
    //     = 4.5
    //

    thrust::device_vector<double> vx{3.0};
    thrust::device_vector<double> vy{3.5};
    thrust::device_vector<double> vz{8.0};

    thrust::device_vector<double> ax{6.0};
    thrust::device_vector<double> ay{2.0};
    thrust::device_vector<double> az{-4.0};

    const double dt = 0.5;

    leapfrogKick(
        thrust::raw_pointer_cast(vx.data()),
        thrust::raw_pointer_cast(vy.data()),
        thrust::raw_pointer_cast(vz.data()),
        thrust::raw_pointer_cast(ax.data()),
        thrust::raw_pointer_cast(ay.data()),
        thrust::raw_pointer_cast(az.data()),
        1,
        dt);

    const auto hostVx = copyToHost(vx);
    const auto hostVy = copyToHost(vy);
    const auto hostVz = copyToHost(vz);

    EXPECT_DOUBLE_EQ(
        hostVx[0],
        4.5);

    EXPECT_DOUBLE_EQ(
        hostVy[0],
        4.0);

    EXPECT_DOUBLE_EQ(
        hostVz[0],
        7.0);
}


TEST_F(
    IntegratorTest,
    ConstantAccelerationMatchesAnalyticalSolution)
{
    //
    // For constant acceleration:
    //
    // x(t + dt)
    // =
    // x(t)
    // + v(t) dt
    // + 1/2 a dt^2
    //
    // v(t + dt)
    // =
    // v(t)
    // + a dt
    //
    // Leapfrog should reproduce this exactly,
    // apart from floating-point roundoff.
    //

    thrust::device_vector<double> x{1.0};
    thrust::device_vector<double> y{-2.0};
    thrust::device_vector<double> z{3.0};

    thrust::device_vector<double> vx{4.0};
    thrust::device_vector<double> vy{5.0};
    thrust::device_vector<double> vz{-6.0};

    thrust::device_vector<double> ax{2.0};
    thrust::device_vector<double> ay{-4.0};
    thrust::device_vector<double> az{1.0};

    const double dt = 0.25;

    leapfrogKickDrift(
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(vx.data()),
        thrust::raw_pointer_cast(vy.data()),
        thrust::raw_pointer_cast(vz.data()),
        thrust::raw_pointer_cast(ax.data()),
        thrust::raw_pointer_cast(ay.data()),
        thrust::raw_pointer_cast(az.data()),
        1,
        dt);

    //
    // Constant acceleration means a_(n+1) = a_n.
    //
    leapfrogKick(
        thrust::raw_pointer_cast(vx.data()),
        thrust::raw_pointer_cast(vy.data()),
        thrust::raw_pointer_cast(vz.data()),
        thrust::raw_pointer_cast(ax.data()),
        thrust::raw_pointer_cast(ay.data()),
        thrust::raw_pointer_cast(az.data()),
        1,
        dt);

    const auto hostX = copyToHost(x);
    const auto hostY = copyToHost(y);
    const auto hostZ = copyToHost(z);

    const auto hostVx = copyToHost(vx);
    const auto hostVy = copyToHost(vy);
    const auto hostVz = copyToHost(vz);

    const double expectedX =
        1.0 +
        4.0 * dt +
        0.5 * 2.0 * dt * dt;

    const double expectedY =
        -2.0 +
        5.0 * dt +
        0.5 * (-4.0) * dt * dt;

    const double expectedZ =
        3.0 +
        (-6.0) * dt +
        0.5 * 1.0 * dt * dt;

    const double expectedVx =
        4.0 + 2.0 * dt;

    const double expectedVy =
        5.0 - 4.0 * dt;

    const double expectedVz =
        -6.0 + 1.0 * dt;

    EXPECT_NEAR(
        hostX[0],
        expectedX,
        1.0e-14);

    EXPECT_NEAR(
        hostY[0],
        expectedY,
        1.0e-14);

    EXPECT_NEAR(
        hostZ[0],
        expectedZ,
        1.0e-14);

    EXPECT_NEAR(
        hostVx[0],
        expectedVx,
        1.0e-14);

    EXPECT_NEAR(
        hostVy[0],
        expectedVy,
        1.0e-14);

    EXPECT_NEAR(
        hostVz[0],
        expectedVz,
        1.0e-14);
}


TEST_F(
    IntegratorTest,
    UpdatesMultipleParticlesIndependently)
{
    thrust::device_vector<double> x{
        0.0,
        10.0,
        -5.0};

    thrust::device_vector<double> y{
        0.0,
        0.0,
        0.0};

    thrust::device_vector<double> z{
        0.0,
        0.0,
        0.0};

    thrust::device_vector<double> vx{
        1.0,
        2.0,
        3.0};

    thrust::device_vector<double> vy{
        0.0,
        0.0,
        0.0};

    thrust::device_vector<double> vz{
        0.0,
        0.0,
        0.0};

    thrust::device_vector<double> ax{
        2.0,
        -4.0,
        6.0};

    thrust::device_vector<double> ay{
        0.0,
        0.0,
        0.0};

    thrust::device_vector<double> az{
        0.0,
        0.0,
        0.0};

    const double dt = 1.0;

    leapfrogKickDrift(
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(vx.data()),
        thrust::raw_pointer_cast(vy.data()),
        thrust::raw_pointer_cast(vz.data()),
        thrust::raw_pointer_cast(ax.data()),
        thrust::raw_pointer_cast(ay.data()),
        thrust::raw_pointer_cast(az.data()),
        3,
        dt);

    const auto hostX = copyToHost(x);
    const auto hostVx = copyToHost(vx);

    //
    // Particle 0:
    //
    // v_half = 1 + 1 = 2
    // x_new  = 0 + 2 = 2
    //
    EXPECT_DOUBLE_EQ(
        hostVx[0],
        2.0);

    EXPECT_DOUBLE_EQ(
        hostX[0],
        2.0);

    //
    // Particle 1:
    //
    // v_half = 2 - 2 = 0
    // x_new  = 10
    //
    EXPECT_DOUBLE_EQ(
        hostVx[1],
        0.0);

    EXPECT_DOUBLE_EQ(
        hostX[1],
        10.0);

    //
    // Particle 2:
    //
    // v_half = 3 + 3 = 6
    // x_new  = -5 + 6 = 1
    //
    EXPECT_DOUBLE_EQ(
        hostVx[2],
        6.0);

    EXPECT_DOUBLE_EQ(
        hostX[2],
        1.0);
}


TEST_F(
    IntegratorTest,
    RejectsInvalidTimeStep)
{
    thrust::device_vector<double> value{0.0};

    double* pointer =
        thrust::raw_pointer_cast(
            value.data());

    EXPECT_THROW(
        leapfrogKickDrift(
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            0.0),
        std::invalid_argument);

    EXPECT_THROW(
        leapfrogKick(
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            -1.0),
        std::invalid_argument);

    EXPECT_THROW(
        leapfrogKick(
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            std::numeric_limits<double>::infinity()),
        std::invalid_argument);
}


TEST_F(
    IntegratorTest,
    RejectsNullPointersForNonEmptySystem)
{
    thrust::device_vector<double> value{0.0};

    double* pointer =
        thrust::raw_pointer_cast(
            value.data());

    EXPECT_THROW(
        leapfrogKickDrift(
            nullptr,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            0.1),
        std::invalid_argument);

    EXPECT_THROW(
        leapfrogKick(
            nullptr,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            1,
            0.1),
        std::invalid_argument);
}


TEST_F(
    IntegratorTest,
    EmptySystemIsNoOp)
{
    EXPECT_NO_THROW(
        leapfrogKickDrift(
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0.1));

    EXPECT_NO_THROW(
        leapfrogKick(
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            0,
            0.1));
}

} // namespace