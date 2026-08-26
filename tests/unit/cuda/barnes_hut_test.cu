#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "barnes_hut.hpp"
#include "cuda_test.hpp"
#include "octree_builder.hpp"
#include "octree_geometry.hpp"
#include "octree_physics.hpp"

namespace
{

template <typename T>
void copyToDevice(
    T* destination,
    const std::vector<T>& source)
{
    const cudaError_t status =
        cudaMemcpy(
            destination,
            source.data(),
            source.size() * sizeof(T),
            cudaMemcpyHostToDevice);

    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(status));
    }
}


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


struct Acceleration
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
};


Acceleration directAcceleration(
    const std::vector<double>& mass,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& z,
    double gravitationalConstant,
    double softening)
{
    const std::size_t count = mass.size();

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
                x[source] - x[target];

            const double dy =
                y[source] - y[target];

            const double dz =
                z[source] - z[target];

            const double distanceSquared =
                dx * dx +
                dy * dy +
                dz * dz +
                softeningSquared;

            const double inverseDistance =
                1.0 / std::sqrt(distanceSquared);

            const double inverseDistanceCubed =
                inverseDistance *
                inverseDistance *
                inverseDistance;

            const double factor =
                gravitationalConstant *
                mass[source] *
                inverseDistanceCubed;

            result.x[target] += factor * dx;
            result.y[target] += factor * dy;
            result.z[target] += factor * dz;
        }
    }

    return result;
}


class BarnesHutTest : public CudaTest
{
protected:
    Octree octree{};

    void TearDown() override
    {
        freeOctree(octree);
    }

    void allocateTree(int nodeCount)
    {
        allocateOctreeTopology(
            octree,
            nodeCount);

        allocateOctreePhysicalData(
            octree);

        allocateOctreeGeometry(
            octree);

        octree.root = 0;
    }
};


TEST_F(
    BarnesHutTest,
    SingleParticleHasZeroAcceleration)
{
    allocateTree(1);

    octree.nParticles = 1;
    octree.nLeaves = 1;

    copyToDevice(
        octree.children,
        std::vector<int>(8, -1));

    copyToDevice(
        octree.firstParticle,
        std::vector<int>{0});

    copyToDevice(
        octree.particleCount,
        std::vector<int>{1});

    copyToDevice(
        octree.mass,
        std::vector<double>{2.0});

    copyToDevice(
        octree.comX,
        std::vector<double>{3.0});

    copyToDevice(
        octree.comY,
        std::vector<double>{4.0});

    copyToDevice(
        octree.comZ,
        std::vector<double>{5.0});

    copyToDevice(
        octree.centerX,
        std::vector<double>{3.0});

    copyToDevice(
        octree.centerY,
        std::vector<double>{4.0});

    copyToDevice(
        octree.centerZ,
        std::vector<double>{5.0});

    copyToDevice(
        octree.halfSize,
        std::vector<double>{1.0});

    thrust::device_vector<std::uint32_t>
        sortedIndices{0u};

    thrust::device_vector<double>
        mass{2.0};

    thrust::device_vector<double>
        x{3.0};

    thrust::device_vector<double>
        y{4.0};

    thrust::device_vector<double>
        z{5.0};

    thrust::device_vector<double>
        ax(1);

    thrust::device_vector<double>
        ay(1);

    thrust::device_vector<double>
        az(1);

    BarnesHutParameters parameters;
    parameters.theta = 0.5;
    parameters.softening = 1.0e-3;
    parameters.gravitationalConstant = 1.0;

    computeBarnesHutAcceleration(
        octree,
        thrust::raw_pointer_cast(
            sortedIndices.data()),
        thrust::raw_pointer_cast(
            mass.data()),
        thrust::raw_pointer_cast(
            x.data()),
        thrust::raw_pointer_cast(
            y.data()),
        thrust::raw_pointer_cast(
            z.data()),
        thrust::raw_pointer_cast(
            ax.data()),
        thrust::raw_pointer_cast(
            ay.data()),
        thrust::raw_pointer_cast(
            az.data()),
        1,
        parameters);

    const auto resultX =
        copyFromDevice(
            thrust::raw_pointer_cast(
                ax.data()),
            1);

    const auto resultY =
        copyFromDevice(
            thrust::raw_pointer_cast(
                ay.data()),
            1);

    const auto resultZ =
        copyFromDevice(
            thrust::raw_pointer_cast(
                az.data()),
            1);

    EXPECT_DOUBLE_EQ(resultX[0], 0.0);
    EXPECT_DOUBLE_EQ(resultY[0], 0.0);
    EXPECT_DOUBLE_EQ(resultZ[0], 0.0);
}


TEST_F(
    BarnesHutTest,
    ThetaZeroMatchesDirectNBody)
{
    //
    // Tree:
    //
    //               node 0
    //              /      \
    //         node 1      node 2
    //                      /   \
    //                 node 3  node 4
    //
    // node 1 -> particle 0
    // node 3 -> particle 1
    // node 4 -> particle 2
    //

    allocateTree(5);

    octree.nParticles = 3;
    octree.nLeaves = 3;

    std::vector<int> children(5 * 8, -1);

    children[0 * 8 + 0] = 1;
    children[0 * 8 + 7] = 2;

    children[2 * 8 + 0] = 3;
    children[2 * 8 + 7] = 4;

    copyToDevice(
        octree.children,
        children);

    copyToDevice(
        octree.firstParticle,
        std::vector<int>{
            -1,
             0,
            -1,
             1,
             2});

    copyToDevice(
        octree.particleCount,
        std::vector<int>{
            0,
            1,
            0,
            1,
            1});

    //
    // Particles:
    //
    // p0 = -10, mass 1
    // p1 =  10, mass 2
    // p2 =  12, mass 3
    //
    const std::vector<double> hostMass{
        1.0,
        2.0,
        3.0};

    const std::vector<double> hostX{
        -10.0,
         10.0,
         12.0};

    const std::vector<double> hostY{
        0.0,
        0.0,
        0.0};

    const std::vector<double> hostZ{
        0.0,
        0.0,
        0.0};

    //
    // Node masses.
    //
    copyToDevice(
        octree.mass,
        std::vector<double>{
            6.0,
            1.0,
            5.0,
            2.0,
            3.0});

    //
    // Center of mass.
    //
    copyToDevice(
        octree.comX,
        std::vector<double>{
            46.0 / 6.0,
            -10.0,
             11.2,
             10.0,
             12.0});

    copyToDevice(
        octree.comY,
        std::vector<double>(
            5,
            0.0));

    copyToDevice(
        octree.comZ,
        std::vector<double>(
            5,
            0.0));

    //
    // Cell geometry.
    //
    copyToDevice(
        octree.centerX,
        std::vector<double>{
             0.0,
            -8.0,
             8.0,
            10.0,
            12.0});

    copyToDevice(
        octree.centerY,
        std::vector<double>(
            5,
            0.0));

    copyToDevice(
        octree.centerZ,
        std::vector<double>(
            5,
            0.0));

    copyToDevice(
        octree.halfSize,
        std::vector<double>{
            16.0,
             8.0,
             8.0,
             1.0,
             1.0});

    thrust::device_vector<std::uint32_t>
        sortedIndices{
            0u,
            1u,
            2u};

    thrust::device_vector<double>
        mass(
            hostMass.begin(),
            hostMass.end());

    thrust::device_vector<double>
        x(
            hostX.begin(),
            hostX.end());

    thrust::device_vector<double>
        y(
            hostY.begin(),
            hostY.end());

    thrust::device_vector<double>
        z(
            hostZ.begin(),
            hostZ.end());

    thrust::device_vector<double> ax(3);
    thrust::device_vector<double> ay(3);
    thrust::device_vector<double> az(3);

    BarnesHutParameters parameters;

    //
    // theta == 0 means:
    //
    // side^2 < theta^2 * distance^2
    //
    // can never be true.
    //
    // Therefore every internal node must be opened.
    //
    parameters.theta = 0.0;
    parameters.softening = 0.1;
    parameters.gravitationalConstant = 1.0;

    computeBarnesHutAcceleration(
        octree,
        thrust::raw_pointer_cast(
            sortedIndices.data()),
        thrust::raw_pointer_cast(
            mass.data()),
        thrust::raw_pointer_cast(
            x.data()),
        thrust::raw_pointer_cast(
            y.data()),
        thrust::raw_pointer_cast(
            z.data()),
        thrust::raw_pointer_cast(
            ax.data()),
        thrust::raw_pointer_cast(
            ay.data()),
        thrust::raw_pointer_cast(
            az.data()),
        3,
        parameters);

    const Acceleration expected =
        directAcceleration(
            hostMass,
            hostX,
            hostY,
            hostZ,
            parameters.gravitationalConstant,
            parameters.softening);

    const auto actualX =
        copyFromDevice(
            thrust::raw_pointer_cast(
                ax.data()),
            3);

    const auto actualY =
        copyFromDevice(
            thrust::raw_pointer_cast(
                ay.data()),
            3);

    const auto actualZ =
        copyFromDevice(
            thrust::raw_pointer_cast(
                az.data()),
            3);

    for (std::size_t particle = 0;
         particle < 3;
         ++particle)
    {
        EXPECT_NEAR(
            actualX[particle],
            expected.x[particle],
            1.0e-12);

        EXPECT_NEAR(
            actualY[particle],
            expected.y[particle],
            1.0e-12);

        EXPECT_NEAR(
            actualZ[particle],
            expected.z[particle],
            1.0e-12);
    }
}


TEST_F(
    BarnesHutTest,
    AcceptsFarClusterAsSingleMass)
{
    allocateTree(5);

    octree.nParticles = 3;
    octree.nLeaves = 3;

    std::vector<int> children(5 * 8, -1);

    children[0 * 8 + 0] = 1;
    children[0 * 8 + 7] = 2;

    children[2 * 8 + 0] = 3;
    children[2 * 8 + 7] = 4;

    copyToDevice(
        octree.children,
        children);

    copyToDevice(
        octree.firstParticle,
        std::vector<int>{
            -1,
             0,
            -1,
             1,
             2});

    copyToDevice(
        octree.particleCount,
        std::vector<int>{
            0,
            1,
            0,
            1,
            1});

    const std::vector<double> hostMass{
        1.0,
        2.0,
        3.0};

    const std::vector<double> hostX{
        -10.0,
         10.0,
         12.0};

    copyToDevice(
        octree.mass,
        std::vector<double>{
            6.0,
            1.0,
            5.0,
            2.0,
            3.0});

    copyToDevice(
        octree.comX,
        std::vector<double>{
            46.0 / 6.0,
            -10.0,
             11.2,
             10.0,
             12.0});

    copyToDevice(
        octree.comY,
        std::vector<double>(5, 0.0));

    copyToDevice(
        octree.comZ,
        std::vector<double>(5, 0.0));

    copyToDevice(
        octree.centerX,
        std::vector<double>{
             0.0,
            -8.0,
             8.0,
            10.0,
            12.0});

    copyToDevice(
        octree.centerY,
        std::vector<double>(5, 0.0));

    copyToDevice(
        octree.centerZ,
        std::vector<double>(5, 0.0));

    copyToDevice(
        octree.halfSize,
        std::vector<double>{
            16.0,
             8.0,
             8.0,
             1.0,
             1.0});

    thrust::device_vector<std::uint32_t>
        sortedIndices{
            0u,
            1u,
            2u};

    thrust::device_vector<double>
        mass{
            1.0,
            2.0,
            3.0};

    thrust::device_vector<double>
        x{
            -10.0,
             10.0,
             12.0};

    thrust::device_vector<double>
        y{
            0.0,
            0.0,
            0.0};

    thrust::device_vector<double>
        z{
            0.0,
            0.0,
            0.0};

    thrust::device_vector<double> ax(3);
    thrust::device_vector<double> ay(3);
    thrust::device_vector<double> az(3);

    BarnesHutParameters parameters;
    parameters.theta = 1.0;
    parameters.softening = 0.0;
    parameters.gravitationalConstant = 1.0;

    computeBarnesHutAcceleration(
        octree,
        thrust::raw_pointer_cast(
            sortedIndices.data()),
        thrust::raw_pointer_cast(
            mass.data()),
        thrust::raw_pointer_cast(
            x.data()),
        thrust::raw_pointer_cast(
            y.data()),
        thrust::raw_pointer_cast(
            z.data()),
        thrust::raw_pointer_cast(
            ax.data()),
        thrust::raw_pointer_cast(
            ay.data()),
        thrust::raw_pointer_cast(
            az.data()),
        3,
        parameters);

    const auto actualX =
        copyFromDevice(
            thrust::raw_pointer_cast(
                ax.data()),
            3);

    //
    // For target particle 0:
    //
    // cluster node 2:
    //
    // mass = 5
    // CoM  = 11.2
    //
    // target = -10
    //
    // distance = 21.2
    //
    // With G = 1 and no softening:
    //
    //     a = M / r^2
    //
    const double expectedApproximation =
        5.0 / (21.2 * 21.2);

    EXPECT_NEAR(
        actualX[0],
        expectedApproximation,
        1.0e-12);

    //
    // Demonstrate that this was genuinely an approximation and not
    // accidental traversal to both source particles.
    //
    const double exactAcceleration =
        2.0 / (20.0 * 20.0) +
        3.0 / (22.0 * 22.0);

    EXPECT_GT(
        std::abs(
            actualX[0] -
            exactAcceleration),
        1.0e-6);
}

} // namespace