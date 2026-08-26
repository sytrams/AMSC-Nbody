#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "barnes_hut.hpp"
#include "radix_to_octree.hpp"

namespace
{
    // During a depth-first traversal, at each level we may leave at most
    // seven siblings on the stack while descending through the eighth child.
    //
    // With a maximum octree depth of 10:
    //
    //     1 + 7 * 10 = 71
    //
    // entries are sufficient for a valid octree.
    constexpr int kTraversalStackCapacity = 1 + 7 * kOctreeMaxLevel;

    void checkCuda(cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }


    const char* traversalErrorMessage(int errorCode)
    {
        switch (errorCode)
        {
            case 1:
                return "Barnes-Hut traversal found an invalid child node";

            case 2:
                return "Barnes-Hut traversal found an invalid leaf particle range";

            case 3:
                return "Sorted particle index is outside the particle array";

            case 4:
                return "Barnes-Hut traversal stack overflow";

            case 5:
                return "Barnes-Hut traversal found an empty terminal node";

            case 6:
                return "Singular gravitational interaction; use positive softening";

            case 7:
                return "Occupied octree leaf unexpectedly has children";

            case 8:
                return "Octree node has invalid geometry";

            default:
                return "Unknown Barnes-Hut traversal error";
        }
    }
} // namespace


__device__ bool accumulatePointMassAcceleration(double targetX, double targetY, double targetZ, double sourceX, double sourceY, double sourceZ, double sourceMass, double gravitationalConstant, double softeningSquared, double& accelerationX, double& accelerationY, double& accelerationZ)
{
    if (sourceMass == 0.0)
        return true;

    const double dx = sourceX - targetX;
    const double dy = sourceY - targetY;
    const double dz = sourceZ - targetZ;

    const double distanceSquared = dx * dx + dy * dy + dz * dz + softeningSquared;

    // This can only occur for coincident points when softening == 0.
    if (distanceSquared <= 0.0)
        return false;

    const double inverseDistance = 1.0 / sqrt(distanceSquared);
    const double inverseDistanceCubed = inverseDistance * inverseDistance * inverseDistance;
    const double factor = gravitationalConstant * sourceMass * inverseDistanceCubed;

    accelerationX += factor * dx;
    accelerationY += factor * dy;
    accelerationZ += factor * dz;

    return true;
}


__device__ bool nodeContainsTarget(const Octree& octree, int node, double targetX, double targetY, double targetZ)
{
    const double halfSize = octree.halfSize[node];

    return fabs(targetX - octree.centerX[node]) <= halfSize && fabs(targetY - octree.centerY[node]) <= halfSize && fabs(targetZ - octree.centerZ[node]) <= halfSize;
}


__global__ void computeBarnesHutAccelerationKernel(Octree octree, const std::uint32_t* sortedIndices, const double* particleMass, const double* positionX, const double* positionY, const double* positionZ, double* accelerationX, double* accelerationY, double* accelerationZ, double theta, double softeningSquared, double gravitationalConstant, int* errorCode)
{
    const int target = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (target >= octree.nParticles)
        return;

    const double targetX = positionX[target];
    const double targetY = positionY[target];
    const double targetZ = positionZ[target];
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    int stack[kTraversalStackCapacity];
    int stackSize = 0;

    stack[stackSize++] = octree.root;

    const double thetaSquared = theta * theta;

    while (stackSize > 0)
    {
        const int node = stack[--stackSize];
        const double halfSize = octree.halfSize[node];

        if (halfSize <= 0.0)
        {
            atomicCAS(errorCode, 0, 8);
            return;
        }

        const int leafParticleCount = octree.particleCount[node];
        
        // Occupied leaf
        // Leaves are always evaluated exactly
        if (leafParticleCount > 0)
        {
            for (int octant = 0; octant < 8; ++octant)
            {
                if (octree.children[node * 8 + octant] != -1)
                {
                    atomicCAS(errorCode, 0, 7);
                    return;
                }
            }

            const int first = octree.firstParticle[node];

            if (first < 0 || leafParticleCount > octree.nParticles || first > octree.nParticles - leafParticleCount)
            {
                atomicCAS(errorCode, 0, 2);
                return;
            }

            for (int local = 0; local < leafParticleCount; ++local)
            {
                const int sortedPosition = first + local;
                const std::uint32_t source = sortedIndices[sortedPosition];

                if (source >= static_cast<std::uint32_t>(octree.nParticles))
                {
                    atomicCAS(errorCode, 0, 3);
                    return;
                }

                // Explicitly remove self interaction.
                if (source == static_cast<std::uint32_t>(target))
                    continue;

                if (!accumulatePointMassAcceleration(targetX, targetY, targetZ, positionX[source], positionY[source], positionZ[source], particleMass[source], gravitationalConstant, softeningSquared, ax, ay, az))
                {
                    atomicCAS(errorCode, 0, 6);
                    return;
                }
            }
            continue;
        }

        // Internal node
        const double nodeMass = octree.mass[node];

        // A zero-mass node cannot contribute to the acceleration.
        if (nodeMass == 0.0)
            continue;

        const double dx = octree.comX[node] - targetX;
        const double dy = octree.comY[node] - targetY;
        const double dz = octree.comZ[node] - targetZ;
        const double distanceSquared = dx * dx + dy * dy + dz * dz;
        const double side = 2.0 * halfSize;
        const bool containsTarget = nodeContainsTarget(octree, node, targetX, targetY, targetZ);

        // Barnes-Hut Multipole Acceptance Criterion
        const bool acceptNode = !containsTarget && distanceSquared > 0.0 && side * side < thetaSquared * distanceSquared;

        if (acceptNode)
        {
            if (!accumulatePointMassAcceleration(targetX, targetY, targetZ, octree.comX[node], octree.comY[node], octree.comZ[node], nodeMass, gravitationalConstant, softeningSquared, ax, ay, az))
            {
                atomicCAS(errorCode, 0, 6);
                return;
            }
            continue;
        }

        int childCount = 0;

        for (int octant = 0; octant < 8; ++octant)
        {
            const int child = octree.children[node * 8 + octant];

            if (child == -1)
                continue;

            // Node zero is the root
            if (child <= 0 || child >= octree.nNodes)
            {
                atomicCAS(errorCode, 0, 1);
                return;
            }

            if (stackSize >= kTraversalStackCapacity)
            {
                atomicCAS(errorCode, 0, 4);
                return;
            }

            stack[stackSize++] = child;
            ++childCount;
        }

        if (childCount == 0)
        {
            atomicCAS(errorCode, 0, 5);
            return;
        }
    }

    // Each CUDA thread owns exactly one target particle, so no atomics are required for acceleration output.
    accelerationX[target] = ax;
    accelerationY[target] = ay;
    accelerationZ[target] = az;
}


void computeBarnesHutAcceleration(const Octree& octree, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ, double* d_accelerationX, double* d_accelerationY, double* d_accelerationZ, std::size_t particleCount, const BarnesHutParameters& parameters)
{
    
    // Validate octree state
    if (octree.nParticles <= 0 || octree.nNodes <= 0 || octree.root != 0)
        throw std::logic_error("Octree is not valid for Barnes-Hut traversal");
    
    if (particleCount != static_cast<std::size_t>(octree.nParticles))
        throw std::invalid_argument("Particle count does not match the octree");
    
    // Topology required by traversal.
    if (octree.children == nullptr || octree.firstParticle == nullptr || octree.particleCount == nullptr)
        throw std::logic_error("Octree topology arrays required for traversal are not allocated");
   
    // Physical data required by Barnes-Hut approximation.
    if (octree.mass == nullptr || octree.comX == nullptr || octree.comY == nullptr || octree.comZ == nullptr)
        throw std::logic_error("Octree physical data is not allocated");
    
    // Cell geometry required by the acceptance criterion.
    if (octree.centerX == nullptr || octree.centerY == nullptr || octree.centerZ == nullptr || octree.halfSize == nullptr)
        throw std::logic_error("Octree geometry is not allocated");
    

    // Particle data.
    if (d_sortedIndices == nullptr || d_mass == nullptr || d_positionX == nullptr || d_positionY == nullptr || d_positionZ == nullptr)
        throw std::invalid_argument("Barnes-Hut particle arrays must not be null");
    
    if (d_accelerationX == nullptr || d_accelerationY == nullptr || d_accelerationZ == nullptr)
        throw std::invalid_argument("Barnes-Hut acceleration arrays must not be null");
    
    // Algorithm parameters.
    if (!std::isfinite(parameters.theta) || parameters.theta < 0.0)
        throw std::invalid_argument("Barnes-Hut theta must be finite and non-negative");
    
    if (!std::isfinite(parameters.softening) || parameters.softening < 0.0)
        throw std::invalid_argument("Barnes-Hut softening must be finite and non-negative");
    
    if (!std::isfinite(parameters.gravitationalConstant) || parameters.gravitationalConstant <= 0.0)
        throw std::invalid_argument("Gravitational constant must be finite and positive");

    const double softeningSquared = parameters.softening * parameters.softening;
    int* errorCodeDevice = nullptr;

    try
    {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&errorCodeDevice), sizeof(int)), "allocate Barnes-Hut error code");
        checkCuda(cudaMemset(errorCodeDevice, 0, sizeof(int)), "initialize Barnes-Hut error code");

        constexpr int threadsPerBlock = 256;
        const int blocks = (octree.nParticles + threadsPerBlock - 1) / threadsPerBlock;

        computeBarnesHutAccelerationKernel<<<blocks, threadsPerBlock>>>(octree, d_sortedIndices, d_mass, d_positionX, d_positionY, d_positionZ, d_accelerationX, d_accelerationY, d_accelerationZ, parameters.theta, softeningSquared, parameters.gravitationalConstant, errorCodeDevice);

        checkCuda(cudaGetLastError(), "computeBarnesHutAccelerationKernel launch");
        checkCuda(cudaDeviceSynchronize(), "computeBarnesHutAccelerationKernel execution");

        int errorCode = 0;

        checkCuda(cudaMemcpy(&errorCode, errorCodeDevice, sizeof(int), cudaMemcpyDeviceToHost), "copy Barnes-Hut error code");

        if (errorCode != 0)
            throw std::runtime_error(traversalErrorMessage(errorCode));
        
        cudaFree(errorCodeDevice);
        errorCodeDevice = nullptr;
    }
    catch (...)
    {
        cudaFree(errorCodeDevice);
        throw;
    }
}