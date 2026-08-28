#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "barnes_hut.hpp"
#include "cuda_test.hpp"
#include "globalbounding.hpp"
#include "morton.hpp"
#include "morton_leaf_groups.hpp"
#include "octree_builder.hpp"
#include "octree_geometry.hpp"
#include "octree_physics.hpp"
#include "radix_to_octree.hpp"
#include "tree_builder.hpp"

namespace
{

struct ParticleData
{
    std::vector<double> mass;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    [[nodiscard]] std::size_t size() const noexcept
    {
        return mass.size();
    }
};


struct Acceleration
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
};


struct ErrorMetrics
{
    double relativeL2 = 0.0;
    double meanRelative = 0.0;
    double maxRelative = 0.0;
};


struct SpatialResources
{
    MortonLeafGroups groups{};
    Tree radixTree{};
    RadixToOctreePlan plan{};
    Octree octree{};

    ~SpatialResources()
    {
        freeOctree(octree);
        freeRadixToOctreePlan(plan);
        freeTree(radixTree);
        freeMortonLeafGroups(groups);
    }
};


template <typename T>
std::vector<T> copyFromDevice(
    const T* source,
    std::size_t count)
{
    std::vector<T> result(count);

    const cudaError_t status =
        cudaMemcpy(
            result.data(),
            source,
            count * sizeof(T),
            cudaMemcpyDeviceToHost);

    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(status));
    }

    return result;
}


Acceleration directAcceleration(
    const ParticleData& particles,
    double gravitationalConstant,
    double softening)
{
    const std::size_t count =
        particles.size();

    Acceleration result{
        std::vector<double>(count, 0.0),
        std::vector<double>(count, 0.0),
        std::vector<double>(count, 0.0)};

    const double softeningSquared =
        softening * softening;

    for (std::size_t target = 0;
         target < count;
         ++target)
    {
        for (std::size_t source = 0;
             source < count;
             ++source)
        {
            if (source == target)
                continue;

            const double dx =
                particles.x[source] -
                particles.x[target];

            const double dy =
                particles.y[source] -
                particles.y[target];

            const double dz =
                particles.z[source] -
                particles.z[target];

            const double distanceSquared =
                dx * dx +
                dy * dy +
                dz * dz +
                softeningSquared;

            const double inverseDistance =
                1.0 / std::sqrt(
                    distanceSquared);

            const double inverseDistanceCubed =
                inverseDistance *
                inverseDistance *
                inverseDistance;

            const double factor =
                gravitationalConstant *
                particles.mass[source] *
                inverseDistanceCubed;

            result.x[target] +=
                factor * dx;

            result.y[target] +=
                factor * dy;

            result.z[target] +=
                factor * dz;
        }
    }

    return result;
}


ErrorMetrics computeErrorMetrics(
    const Acceleration& approximate,
    const Acceleration& exact)
{
    const std::size_t count =
        exact.x.size();

    double errorSquaredSum = 0.0;
    double referenceSquaredSum = 0.0;

    for (std::size_t particle = 0;
         particle < count;
         ++particle)
    {
        const double ex =
            approximate.x[particle] -
            exact.x[particle];

        const double ey =
            approximate.y[particle] -
            exact.y[particle];

        const double ez =
            approximate.z[particle] -
            exact.z[particle];

        errorSquaredSum +=
            ex * ex +
            ey * ey +
            ez * ez;

        referenceSquaredSum +=
            exact.x[particle] *
                exact.x[particle] +
            exact.y[particle] *
                exact.y[particle] +
            exact.z[particle] *
                exact.z[particle];
    }

    const double referenceRms =
        std::sqrt(
            referenceSquaredSum /
            static_cast<double>(count));

    /*
     * Avoid unstable particle-wise relative errors
     * when an exact acceleration is extremely close
     * to zero because of force cancellation.
     */
    const double denominatorFloor =
        std::max(
            referenceRms * 1.0e-12,
            1.0e-30);

    double relativeSum = 0.0;
    double maxRelative = 0.0;

    for (std::size_t particle = 0;
         particle < count;
         ++particle)
    {
        const double ex =
            approximate.x[particle] -
            exact.x[particle];

        const double ey =
            approximate.y[particle] -
            exact.y[particle];

        const double ez =
            approximate.z[particle] -
            exact.z[particle];

        const double errorNorm =
            std::sqrt(
                ex * ex +
                ey * ey +
                ez * ez);

        const double referenceNorm =
            std::sqrt(
                exact.x[particle] *
                    exact.x[particle] +
                exact.y[particle] *
                    exact.y[particle] +
                exact.z[particle] *
                    exact.z[particle]);

        const double relative =
            errorNorm /
            std::max(
                referenceNorm,
                denominatorFloor);

        relativeSum += relative;

        if (relative > maxRelative)
            maxRelative = relative;
    }

    ErrorMetrics metrics;

    metrics.relativeL2 =
        std::sqrt(
            errorSquaredSum /
            referenceSquaredSum);

    metrics.meanRelative =
        relativeSum /
        static_cast<double>(count);

    metrics.maxRelative =
        maxRelative;

    return metrics;
}


ParticleData makeUniformParticles(
    std::size_t count)
{
    std::mt19937_64 generator(
        0xBADC0FFEEULL);

    std::uniform_real_distribution<double>
        positionDistribution(
            -1.0,
             1.0);

    std::uniform_real_distribution<double>
        massDistribution(
            0.5,
            2.0);

    ParticleData particles;

    particles.mass.resize(count);
    particles.x.resize(count);
    particles.y.resize(count);
    particles.z.resize(count);

    for (std::size_t particle = 0;
         particle < count;
         ++particle)
    {
        particles.mass[particle] =
            massDistribution(generator);

        particles.x[particle] =
            positionDistribution(generator);

        particles.y[particle] =
            positionDistribution(generator);

        particles.z[particle] =
            positionDistribution(generator);
    }

    return particles;
}


ParticleData makeClusteredParticles(
    std::size_t count)
{
    std::mt19937_64 generator(
        0xC1A57E5ULL);

    std::normal_distribution<double>
        offsetDistribution(
            0.0,
            0.08);

    std::uniform_real_distribution<double>
        massDistribution(
            0.5,
            2.0);

    constexpr std::array<
        std::array<double, 3>,
        4>
        centers{{
            {-0.70, -0.65, -0.60},
            { 0.70,  0.65,  0.60},
            {-0.60,  0.70, -0.55},
            { 0.65, -0.70,  0.70}
        }};

    ParticleData particles;

    particles.mass.resize(count);
    particles.x.resize(count);
    particles.y.resize(count);
    particles.z.resize(count);

    for (std::size_t particle = 0;
         particle < count;
         ++particle)
    {
        const auto& center =
            centers[
                particle %
                centers.size()];

        particles.mass[particle] =
            massDistribution(generator);

        particles.x[particle] =
            center[0] +
            offsetDistribution(generator);

        particles.y[particle] =
            center[1] +
            offsetDistribution(generator);

        particles.z[particle] =
            center[2] +
            offsetDistribution(generator);
    }

    return particles;
}


class BarnesHutAccuracyTest : public CudaTest
{
protected:

    void runCharacterization(
        const ParticleData& particles,
        const std::string& datasetName)
    {
        ASSERT_GT(
            particles.size(),
            1u);

        ASSERT_EQ(
            particles.mass.size(),
            particles.x.size());

        ASSERT_EQ(
            particles.x.size(),
            particles.y.size());

        ASSERT_EQ(
            particles.y.size(),
            particles.z.size());

        const std::size_t particleCount =
            particles.size();

        const int particleCountInt =
            static_cast<int>(
                particleCount);

        thrust::device_vector<double>
            dMass(
                particles.mass.begin(),
                particles.mass.end());

        thrust::device_vector<double>
            dX(
                particles.x.begin(),
                particles.x.end());

        thrust::device_vector<double>
            dY(
                particles.y.begin(),
                particles.y.end());

        thrust::device_vector<double>
            dZ(
                particles.z.begin(),
                particles.z.end());

        /*
         * Build exactly the spatial pipeline used
         * by the real simulation.
         */
        Bbox boundingBox(
            thrust::raw_pointer_cast(
                dX.data()),
            thrust::raw_pointer_cast(
                dY.data()),
            thrust::raw_pointer_cast(
                dZ.data()),
            particleCount);

        MortonKeys morton(
            thrust::raw_pointer_cast(
                dX.data()),
            thrust::raw_pointer_cast(
                dY.data()),
            thrust::raw_pointer_cast(
                dZ.data()),
            particleCount,
            boundingBox);

        SpatialResources resources;

        allocateMortonLeafGroups(
            resources.groups,
            particleCountInt);

        buildMortonLeafGroups(
            resources.groups,
            morton.keys_device_data(),
            particleCountInt);

        ASSERT_GT(
            resources.groups.nGroups,
            0);

        allocateTree(
            resources.radixTree,
            resources.groups.nGroups);

        buildTreeFromMortonGroups(
            resources.radixTree,
            resources.groups);

        allocateRadixToOctreePlan(
            resources.plan,
            2 * resources.groups.nGroups - 1);

        buildRadixToOctreePlan(
            resources.plan,
            resources.radixTree,
            resources.groups);

        allocateOctreeTopology(
            resources.octree,
            resources.plan.nOctreeNodes);

        buildSparseOctreeTopology(
            resources.octree,
            resources.radixTree,
            resources.groups,
            resources.plan);

        allocateOctreeGeometry(
            resources.octree);

        computeOctreeGeometry(
            resources.octree,
            boundingBox);

        allocateOctreePhysicalData(
            resources.octree);

        computeOctreeMassAndCenterOfMass(
            resources.octree,
            morton.indices_device_data(),
            thrust::raw_pointer_cast(
                dMass.data()),
            thrust::raw_pointer_cast(
                dX.data()),
            thrust::raw_pointer_cast(
                dY.data()),
            thrust::raw_pointer_cast(
                dZ.data()));

        constexpr double gravitationalConstant =
            1.0;

        /*
         * Positive softening avoids singular or
         * nearly singular interactions dominating
         * this Barnes-Hut accuracy experiment.
         */
        constexpr double softening =
            0.02;

        const Acceleration exact =
            directAcceleration(
                particles,
                gravitationalConstant,
                softening);

        thrust::device_vector<double>
            dAx(particleCount);

        thrust::device_vector<double>
            dAy(particleCount);

        thrust::device_vector<double>
            dAz(particleCount);

        constexpr std::array<double, 5>
            thetaValues{
                0.0,
                0.3,
                0.5,
                0.7,
                1.0};

        std::cout
            << "\nBarnes-Hut accuracy: "
            << datasetName
            << '\n';

        std::cout
            << "theta"
            << '\t'
            << "relative_L2"
            << '\t'
            << "mean_relative"
            << '\t'
            << "max_relative"
            << '\n';

        for (const double theta :
             thetaValues)
        {
            BarnesHutParameters parameters;

            parameters.theta =
                theta;

            parameters.softening =
                softening;

            parameters.gravitationalConstant =
                gravitationalConstant;

            computeBarnesHutAcceleration(
                resources.octree,
                morton.indices_device_data(),
                thrust::raw_pointer_cast(
                    dMass.data()),
                thrust::raw_pointer_cast(
                    dX.data()),
                thrust::raw_pointer_cast(
                    dY.data()),
                thrust::raw_pointer_cast(
                    dZ.data()),
                thrust::raw_pointer_cast(
                    dAx.data()),
                thrust::raw_pointer_cast(
                    dAy.data()),
                thrust::raw_pointer_cast(
                    dAz.data()),
                particleCount,
                parameters);

            const Acceleration approximate{
                copyFromDevice(
                    thrust::raw_pointer_cast(
                        dAx.data()),
                    particleCount),

                copyFromDevice(
                    thrust::raw_pointer_cast(
                        dAy.data()),
                    particleCount),

                copyFromDevice(
                    thrust::raw_pointer_cast(
                        dAz.data()),
                    particleCount)};

            const ErrorMetrics metrics =
                computeErrorMetrics(
                    approximate,
                    exact);

            EXPECT_TRUE(
                std::isfinite(
                    metrics.relativeL2));

            EXPECT_TRUE(
                std::isfinite(
                    metrics.meanRelative));

            EXPECT_TRUE(
                std::isfinite(
                    metrics.maxRelative));

            /*
            * Numerical regression bounds.
            *
            * These limits deliberately contain margin above the
            * measured deterministic values. They are intended to
            * detect significant degradation of the Barnes-Hut
            * approximation, not harmless floating-point changes.
            */
            if (theta == 0.0)
            {
                EXPECT_LT(
                    metrics.relativeL2,
                    1.0e-11);
            }
            else if (theta == 0.3)
            {
                EXPECT_LT(
                    metrics.relativeL2,
                    2.0e-3);
            }
            else if (theta == 0.5)
            {
                /*
                * This is the simulation default.
                *
                * Require less than 1% global relative L2 error
                * for both uniform and clustered test problems.
                */
                EXPECT_LT(
                    metrics.relativeL2,
                    1.0e-2);
            }
            else if (theta == 0.7)
            {
                EXPECT_LT(
                    metrics.relativeL2,
                    2.0e-2);
            }
            else if (theta == 1.0)
            {
                EXPECT_LT(
                    metrics.relativeL2,
                    4.0e-2);
            }

            std::cout
                << std::scientific
                << std::setprecision(6)
                << theta
                << '\t'
                << metrics.relativeL2
                << '\t'
                << metrics.meanRelative
                << '\t'
                << metrics.maxRelative
                << '\n';
        }
    }
};


TEST_F(
    BarnesHutAccuracyTest,
    CharacterizesUniformDistribution)
{
    runCharacterization(
        makeUniformParticles(256),
        "uniform-256");
}


TEST_F(
    BarnesHutAccuracyTest,
    CharacterizesClusteredDistribution)
{
    runCharacterization(
        makeClusteredParticles(256),
        "clustered-256");
}

} // namespace