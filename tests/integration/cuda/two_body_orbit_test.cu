#include <gtest/gtest.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "barnes_hut.hpp"
#include "cuda_test.hpp"
#include "globalbounding.hpp"
#include "integrator.hpp"
#include "morton.hpp"
#include "morton_leaf_groups.hpp"
#include "octree_builder.hpp"
#include "octree_geometry.hpp"
#include "octree_physics.hpp"
#include "radix_to_octree.hpp"
#include "tree_builder.hpp"

namespace
{

struct PipelineResources
{
    MortonLeafGroups groups{};
    Tree radixTree{};
    RadixToOctreePlan plan{};
    Octree octree{};

    ~PipelineResources()
    {
        freeOctree(octree);
        freeRadixToOctreePlan(plan);
        freeTree(radixTree);
        freeMortonLeafGroups(groups);
    }
};


double* devicePointer(
    thrust::device_vector<double>& vector)
{
    return thrust::raw_pointer_cast(
        vector.data());
}


const double* devicePointer(
    const thrust::device_vector<double>& vector)
{
    return thrust::raw_pointer_cast(
        vector.data());
}


std::vector<double> copyToHost(
    const thrust::device_vector<double>& device)
{
    std::vector<double> host(device.size());

    thrust::copy(
        device.begin(),
        device.end(),
        host.begin());

    return host;
}


void computeAcceleration(
    const thrust::device_vector<double>& mass,
    const thrust::device_vector<double>& x,
    const thrust::device_vector<double>& y,
    const thrust::device_vector<double>& z,
    thrust::device_vector<double>& ax,
    thrust::device_vector<double>& ay,
    thrust::device_vector<double>& az)
{
    const std::size_t particleCount =
        mass.size();

    //
    // Bounding cube for the current particle configuration.
    //
    Bbox boundingBox(
        devicePointer(x),
        devicePointer(y),
        devicePointer(z),
        particleCount);

    //
    // Morton keys + sorted original-particle indices.
    //
    MortonKeys morton(
        devicePointer(x),
        devicePointer(y),
        devicePointer(z),
        particleCount);

    PipelineResources resources;

    const int count =
        static_cast<int>(particleCount);

    //
    // Group duplicate Morton keys.
    //
    allocateMortonLeafGroups(
        resources.groups,
        count);

    buildMortonLeafGroups(
        resources.groups,
        morton.keys_device_data(),
        count);

    //
    // Karras radix tree.
    //
    allocateTree(
        resources.radixTree,
        resources.groups.nGroups);

    buildTreeFromMortonGroups(
        resources.radixTree,
        resources.groups);

    //
    // Plan conversion radix tree -> sparse octree.
    //
    allocateRadixToOctreePlan(
        resources.plan,
        2 * resources.groups.nGroups - 1);

    buildRadixToOctreePlan(
        resources.plan,
        resources.radixTree,
        resources.groups);

    //
    // Materialize sparse octree topology.
    //
    allocateOctreeTopology(
        resources.octree,
        resources.plan.nOctreeNodes);

    buildSparseOctreeTopology(
        resources.octree,
        resources.radixTree,
        resources.groups,
        resources.plan);

    //
    // Physical data and geometric cells.
    //
    allocateOctreePhysicalData(
        resources.octree);

    allocateOctreeGeometry(
        resources.octree);

    computeOctreeMassAndCenterOfMass(
        resources.octree,
        morton.indices_device_data(),
        devicePointer(mass),
        devicePointer(x),
        devicePointer(y),
        devicePointer(z));

    computeOctreeGeometry(
        resources.octree,
        boundingBox);

    //
    // For this numerical validation we use normalized units:
    //
    // G = 1.
    //
    BarnesHutParameters parameters;
    parameters.theta = 0.5;
    parameters.softening = 0.0;
    parameters.gravitationalConstant = 1.0;

    computeBarnesHutAcceleration(
        resources.octree,
        morton.indices_device_data(),
        devicePointer(mass),
        devicePointer(x),
        devicePointer(y),
        devicePointer(z),
        devicePointer(ax),
        devicePointer(ay),
        devicePointer(az),
        particleCount,
        parameters);
}


struct HostState
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    std::vector<double> vx;
    std::vector<double> vy;
    std::vector<double> vz;
};


HostState copyState(
    const thrust::device_vector<double>& x,
    const thrust::device_vector<double>& y,
    const thrust::device_vector<double>& z,
    const thrust::device_vector<double>& vx,
    const thrust::device_vector<double>& vy,
    const thrust::device_vector<double>& vz)
{
    return {
        copyToHost(x),
        copyToHost(y),
        copyToHost(z),
        copyToHost(vx),
        copyToHost(vy),
        copyToHost(vz)};
}


double separation(
    const HostState& state)
{
    const double dx =
        state.x[1] - state.x[0];

    const double dy =
        state.y[1] - state.y[0];

    const double dz =
        state.z[1] - state.z[0];

    return std::sqrt(
        dx * dx +
        dy * dy +
        dz * dz);
}


double totalEnergy(
    const HostState& state)
{
    constexpr double G = 1.0;
    constexpr double mass0 = 1.0;
    constexpr double mass1 = 1.0;

    const double velocitySquared0 =
        state.vx[0] * state.vx[0] +
        state.vy[0] * state.vy[0] +
        state.vz[0] * state.vz[0];

    const double velocitySquared1 =
        state.vx[1] * state.vx[1] +
        state.vy[1] * state.vy[1] +
        state.vz[1] * state.vz[1];

    const double kineticEnergy =
        0.5 * mass0 * velocitySquared0 +
        0.5 * mass1 * velocitySquared1;

    const double potentialEnergy =
        -G *
        mass0 *
        mass1 /
        separation(state);

    return
        kineticEnergy +
        potentialEnergy;
}


class TwoBodyOrbitTest : public CudaTest
{
};

struct OrbitError
{
    double position;
    double velocity;
};

OrbitError runQuarterOrbit(
    int numberOfSteps)
{
    const double orbitalVelocity =
        1.0 / std::sqrt(2.0);

    thrust::device_vector<double> mass{
        1.0,
        1.0};

    thrust::device_vector<double> x{
        -0.5,
         0.5};

    thrust::device_vector<double> y{
        0.0,
        0.0};

    thrust::device_vector<double> z{
        0.0,
        0.0};

    thrust::device_vector<double> vx{
        0.0,
        0.0};

    thrust::device_vector<double> vy{
        -orbitalVelocity,
         orbitalVelocity};

    thrust::device_vector<double> vz{
        0.0,
        0.0};

    thrust::device_vector<double> ax(2);
    thrust::device_vector<double> ay(2);
    thrust::device_vector<double> az(2);

    const double orbitalPeriod =
        2.0 *
        std::acos(-1.0) /
        std::sqrt(2.0);

    //
    // We integrate only one quarter of the orbit.
    //
    // This avoids measuring the error at a special point where
    // periodic cancellation after a complete orbit could hide part
    // of the integration error.
    //
    const double finalTime =
        orbitalPeriod / 4.0;

    const double dt =
        finalTime /
        static_cast<double>(
            numberOfSteps);

    computeAcceleration(
        mass,
        x,
        y,
        z,
        ax,
        ay,
        az);

    for (int step = 0;
         step < numberOfSteps;
         ++step)
    {
        leapfrogKickDrift(
            devicePointer(x),
            devicePointer(y),
            devicePointer(z),
            devicePointer(vx),
            devicePointer(vy),
            devicePointer(vz),
            devicePointer(ax),
            devicePointer(ay),
            devicePointer(az),
            2,
            dt);

        computeAcceleration(
            mass,
            x,
            y,
            z,
            ax,
            ay,
            az);

        leapfrogKick(
            devicePointer(vx),
            devicePointer(vy),
            devicePointer(vz),
            devicePointer(ax),
            devicePointer(ay),
            devicePointer(az),
            2,
            dt);
    }

    const HostState state =
        copyState(
            x,
            y,
            z,
            vx,
            vy,
            vz);

    //
    // ============================================================
    // Exact state after T / 4
    // ============================================================
    //
    // Initial state:
    //
    // particle 0 = (-0.5, 0)
    // particle 1 = ( 0.5, 0)
    //
    // After one quarter of a counter-clockwise orbit:
    //
    // particle 0 = (0, -0.5)
    // particle 1 = (0,  0.5)
    //
    // and the velocities are:
    //
    // particle 0 = (+v, 0)
    // particle 1 = (-v, 0)
    //
    const double positionErrorSquared =
        state.x[0] * state.x[0] +
        (state.y[0] + 0.5) *
            (state.y[0] + 0.5) +
        state.z[0] * state.z[0] +

        state.x[1] * state.x[1] +
        (state.y[1] - 0.5) *
            (state.y[1] - 0.5) +
        state.z[1] * state.z[1];

    const double velocityErrorSquared =
        (state.vx[0] - orbitalVelocity) *
            (state.vx[0] - orbitalVelocity) +
        state.vy[0] * state.vy[0] +
        state.vz[0] * state.vz[0] +

        (state.vx[1] + orbitalVelocity) *
            (state.vx[1] + orbitalVelocity) +
        state.vy[1] * state.vy[1] +
        state.vz[1] * state.vz[1];

    return {
        std::sqrt(positionErrorSquared),
        std::sqrt(velocityErrorSquared)};
}

TEST_F(
    TwoBodyOrbitTest,
    CharacterizesLeapfrogConvergence)
{
    constexpr std::array<int, 4> stepCounts{
        32,
        64,
        128,
        256};

    std::array<OrbitError, stepCounts.size()>
        errors{};

    std::cout
        << "\nLeapfrog convergence: two-body quarter orbit\n"
        << "steps"
        << '\t'
        << "position_error"
        << '\t'
        << "velocity_error"
        << '\t'
        << "position_ratio"
        << '\t'
        << "velocity_ratio"
        << '\t'
        << "position_order"
        << '\t'
        << "velocity_order"
        << '\n';

    for (std::size_t index = 0;
         index < stepCounts.size();
         ++index)
    {
        errors[index] =
            runQuarterOrbit(
                stepCounts[index]);

        double positionRatio = 0.0;
        double velocityRatio = 0.0;

        double positionOrder = 0.0;
        double velocityOrder = 0.0;

        if (index > 0)
        {
            positionRatio =
                errors[index - 1].position /
                errors[index].position;

            velocityRatio =
                errors[index - 1].velocity /
                errors[index].velocity;

            positionOrder =
                std::log2(positionRatio);

            velocityOrder =
                std::log2(velocityRatio);

            /*
            * Refining the timestep must improve
            * the numerical solution.
            */
            EXPECT_LT(
                errors[index].position,
                errors[index - 1].position);

            EXPECT_LT(
                errors[index].velocity,
                errors[index - 1].velocity);

            /*
            * Leapfrog / Kick-Drift-Kick is a
            * second-order integration method.
            *
            * Therefore:
            *
            *     error(dt) ~ C dt^2
            *
            * and halving dt should reduce the error
            * approximately by a factor of four:
            *
            *     error(dt) / error(dt/2) ~ 4
            *
            * Hence:
            *
            *     p = log2(error(dt) / error(dt/2))
            *
            * should be approximately 2.
            */
            EXPECT_GT(
                positionOrder,
                1.8);

            EXPECT_LT(
                positionOrder,
                2.2);

            EXPECT_GT(
                velocityOrder,
                1.8);

            EXPECT_LT(
                velocityOrder,
                2.2);
        }

        EXPECT_TRUE(
            std::isfinite(
                errors[index].position));

        EXPECT_TRUE(
            std::isfinite(
                errors[index].velocity));

        std::cout
            << stepCounts[index]
            << '\t'
            << std::scientific
            << std::setprecision(8)
            << errors[index].position
            << '\t'
            << errors[index].velocity
            << '\t'
            << positionRatio
            << '\t'
            << velocityRatio
            << '\t'
            << positionOrder
            << '\t'
            << velocityOrder
            << '\n';
    }
}

TEST_F(
    TwoBodyOrbitTest,
    CircularOrbitRemainsStableForOnePeriod)
{
    //
    // ============================================================
    // Normalized two-body problem
    // ============================================================
    //
    // m1 = m2 = 1
    // G  = 1
    //
    // Initial separation:
    //
    //       r = 1
    //
    // Each particle therefore moves around the center of mass
    // with orbital radius:
    //
    //       R = 0.5
    //
    // Gravitational acceleration:
    //
    //       a = G m / r^2 = 1
    //
    // Circular-orbit condition:
    //
    //       v^2 / R = a
    //
    // therefore:
    //
    //       v = sqrt(0.5)
    //
    const double orbitalVelocity =
        1.0 / std::sqrt(2.0);

    thrust::device_vector<double> mass{
        1.0,
        1.0};

    thrust::device_vector<double> x{
        -0.5,
         0.5};

    thrust::device_vector<double> y{
        0.0,
        0.0};

    thrust::device_vector<double> z{
        0.0,
        0.0};

    //
    // Counter-clockwise orbit.
    //
    thrust::device_vector<double> vx{
        0.0,
        0.0};

    thrust::device_vector<double> vy{
        -orbitalVelocity,
         orbitalVelocity};

    thrust::device_vector<double> vz{
        0.0,
        0.0};

    thrust::device_vector<double> ax(2);
    thrust::device_vector<double> ay(2);
    thrust::device_vector<double> az(2);

    const HostState initialState =
        copyState(
            x,
            y,
            z,
            vx,
            vy,
            vz);

    const double initialEnergy =
        totalEnergy(initialState);

    //
    // Angular velocity:
    //
    // omega = v / R
    //       = sqrt(2)
    //
    // Therefore:
    //
    // T = 2 pi / sqrt(2)
    //
    const double orbitalPeriod =
        2.0 *
        std::acos(-1.0) /
        std::sqrt(2.0);

    //
    // 128 timesteps per orbit.
    //
    // This is deliberately not extremely small:
    // the test should verify that second-order Leapfrog behaves
    // correctly at a realistic finite timestep.
    //
    constexpr int numberOfSteps = 128;

    const double dt =
        orbitalPeriod /
        static_cast<double>(numberOfSteps);

    //
    // Initial acceleration a(t0).
    //
    computeAcceleration(
        mass,
        x,
        y,
        z,
        ax,
        ay,
        az);

    double maximumRelativeEnergyError = 0.0;
    double maximumSeparationError = 0.0;

    for (int step = 0;
         step < numberOfSteps;
         ++step)
    {
        //
        // --------------------------------------------------------
        // First half kick + drift
        // --------------------------------------------------------
        //
        leapfrogKickDrift(
            devicePointer(x),
            devicePointer(y),
            devicePointer(z),
            devicePointer(vx),
            devicePointer(vy),
            devicePointer(vz),
            devicePointer(ax),
            devicePointer(ay),
            devicePointer(az),
            2,
            dt);

        //
        // New acceleration:
        //
        // a_(n+1) = a(x_(n+1))
        //
        // This rebuilds the complete spatial hierarchy using the new
        // positions.
        //
        computeAcceleration(
            mass,
            x,
            y,
            z,
            ax,
            ay,
            az);

        //
        // --------------------------------------------------------
        // Second half kick
        // --------------------------------------------------------
        //
        leapfrogKick(
            devicePointer(vx),
            devicePointer(vy),
            devicePointer(vz),
            devicePointer(ax),
            devicePointer(ay),
            devicePointer(az),
            2,
            dt);

        //
        // Monitor numerical invariants throughout the orbit,
        // not only at the final timestep.
        //
        const HostState currentState =
            copyState(
                x,
                y,
                z,
                vx,
                vy,
                vz);

        const double currentEnergy =
            totalEnergy(currentState);

        const double relativeEnergyError =
            std::abs(
                (currentEnergy - initialEnergy) /
                initialEnergy);

        maximumRelativeEnergyError =
            std::max(
                maximumRelativeEnergyError,
                relativeEnergyError);

        maximumSeparationError =
            std::max(
                maximumSeparationError,
                std::abs(
                    separation(currentState) -
                    1.0));
    }

    const HostState finalState =
        copyState(
            x,
            y,
            z,
            vx,
            vy,
            vz);

    //
    // ============================================================
    // 1. Orbit remains approximately circular
    // ============================================================
    //
    EXPECT_LT(
        maximumSeparationError,
        2.0e-3);

    //
    // ============================================================
    // 2. Energy remains bounded
    // ============================================================
    //
    // Leapfrog is symplectic, so we expect bounded oscillation of
    // the energy rather than secular drift.
    //
    EXPECT_LT(
        maximumRelativeEnergyError,
        5.0e-6);

    //
    // ============================================================
    // 3. After one period we return close to the initial position
    // ============================================================
    //
    EXPECT_NEAR(
        finalState.x[0],
        initialState.x[0],
        3.0e-3);

    EXPECT_NEAR(
        finalState.y[0],
        initialState.y[0],
        3.0e-3);

    EXPECT_NEAR(
        finalState.x[1],
        initialState.x[1],
        3.0e-3);

    EXPECT_NEAR(
        finalState.y[1],
        initialState.y[1],
        3.0e-3);

    //
    // ============================================================
    // 4. Velocity also returns close to its initial value
    // ============================================================
    //
    EXPECT_NEAR(
        finalState.vx[0],
        initialState.vx[0],
        4.0e-3);

    EXPECT_NEAR(
        finalState.vy[0],
        initialState.vy[0],
        4.0e-3);

    EXPECT_NEAR(
        finalState.vx[1],
        initialState.vx[1],
        4.0e-3);

    EXPECT_NEAR(
        finalState.vy[1],
        initialState.vy[1],
        4.0e-3);
}

} // namespace