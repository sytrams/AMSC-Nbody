#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include <thrust/device_vector.h>

#include "octree_physics.hpp"

namespace 
{
    void checkCuda(cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }

    const char* physicsErrorMessage(int errorCode)
    {
        switch (errorCode)
        {
            case 1:
                return "Octree contains an invalid child ID";

            case 2:
                return "Occupied octree leaf has invalid particle range";

            case 3:
                return "Sorted particle index is outside the particle array";

            case 4:
                return "Occupied octree leaf has children";

            case 5:
                return "Invalid octree parent during bottom-up propagation";

            case 6:
                return "Invalid octree child count";

            case 7:
                return "An octree parent was signalled too many times";

            default:
                return "Unknown octree physics error";
        }
    }

    int readDeviceError(const int* errorCodeDevice)
    {
        int errorCode = 0;
        checkCuda(cudaMemcpy(&errorCode, errorCodeDevice, sizeof(int), cudaMemcpyDeviceToHost), "copy octree physics error code");
        return errorCode;
    }

    void throwIfDeviceError(const int* errorCodeDevice)
    {
        const int errorCode = readDeviceError(errorCodeDevice);

        if (errorCode != 0)
            throw std::runtime_error(physicsErrorMessage(errorCode));
    }
} // namespace

__global__ void countOctreeChildrenKernel(OctreeDeviceView octree, int* errorCode)
{
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (node >= octree.nNodes)
        return;

    int count = 0;

    for (int octant = 0; octant < 8; ++octant)
    {
        const int child = octree.children[node * 8 + octant];

        if (child == -1)
            continue;

        if (child <= 0 || child >= octree.nNodes)
        {
            atomicCAS(errorCode, 0, 1);
            return;
        }

        ++count;
    }

    octree.childCount[node] = count;

    if (octree.particleCount[node] > 0 && count != 0)
        atomicCAS(errorCode, 0, 4);
}

__global__ void initializeOctreeLeavesKernel(OctreeDeviceView octree, const std::uint32_t* sortedIndices, const double* particleMass, const double* positionX, const double* positionY, const double* positionZ, int* errorCode)
{
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (node >= octree.nNodes)
        return;

    const int count = octree.particleCount[node];

    if (count <= 0)
        return;

    const int first = octree.firstParticle[node];

    if (first < 0 || count <= 0 || first + count > octree.nParticles)
    {
        atomicCAS(errorCode, 0, 2);

        return;
    }

    double totalMass = 0.0f;
    double weightedX = 0.0f;
    double weightedY = 0.0f;
    double weightedZ = 0.0f;

    for (int localParticle = 0; localParticle < count; ++localParticle)
    {
        const int sortedPosition = first + localParticle;
        const std::uint32_t particle = sortedIndices[sortedPosition];

        if (particle >= static_cast<std::uint32_t>(octree.nParticles))
        {
            atomicCAS(errorCode, 0, 3);
            return;
        }

        const double mass = particleMass[particle];
        totalMass += mass;
        weightedX += mass * positionX[particle];
        weightedY += mass * positionY[particle];
        weightedZ += mass * positionZ[particle];
    }

    octree.mass[node] = totalMass;

    if (totalMass > 0.0f)
    {
        const double inverseMass = 1.0f / totalMass;
        octree.comX[node] = weightedX * inverseMass;
        octree.comY[node] = weightedY * inverseMass;
        octree.comZ[node] = weightedZ * inverseMass;
    }
    else
    {
        octree.comX[node] = 0.0f;
        octree.comY[node] = 0.0f;
        octree.comZ[node] = 0.0f;
    }
}

__global__ void propagateOctreeMassKernel(OctreeDeviceView octree, int* errorCode)
{
    const int leaf = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (leaf >= octree.nNodes)
        return;

    //Only occupied octree leaves start a bottom-up propagation.
    if (octree.particleCount[leaf] <= 0)
        return;

    if (octree.childCount[leaf] != 0)
    {
        atomicCAS( errorCode, 0, 4);
        return;
    }

    int completedNode = leaf;
    int parentNode = octree.parent[completedNode];

    while (parentNode != -1)
    {
        if (parentNode < 0 || parentNode >= octree.nNodes)
        {
            atomicCAS(errorCode, 0, 5);
            return;
        }

        const int requiredChildren = octree.childCount[parentNode];

        if (requiredChildren <= 0 || requiredChildren > 8)
        {
            atomicCAS( errorCode, 0, 6);
            return;
        }

        //mass/CoM of completedNode must be visible before the thread announces completion.
        __threadfence();

        const int completedChildren = atomicAdd(&octree.pendingChildren[parentNode], 1) + 1;

        if (completedChildren < requiredChildren)
            return;

        if (completedChildren > requiredChildren)
        {
            atomicCAS( errorCode, 0, 7);
            return;
        }

        //This is the last completed child, so all children of parentNode now have valid mass and CoM
        double totalMass = 0.0f;
        double weightedX = 0.0f;
        double weightedY = 0.0f;
        double weightedZ = 0.0f;

        for (int octant = 0; octant < 8; ++octant)
        {
            const int child = octree.children[parentNode * 8 + octant];

            if (child == -1)
                continue;

            const double childMass = octree.mass[child];
            totalMass += childMass;
            weightedX += childMass * octree.comX[child];
            weightedY += childMass * octree.comY[child];
            weightedZ += childMass * octree.comZ[child];
        }

        octree.mass[parentNode] = totalMass;

        if (totalMass > 0.0f)
        {
            const double inverseMass = 1.0f / totalMass;
            octree.comX[parentNode] = weightedX * inverseMass;
            octree.comY[parentNode] = weightedY * inverseMass;
            octree.comZ[parentNode] = weightedZ * inverseMass;
        }
        else
        {
            octree.comX[parentNode] = 0.0f;
            octree.comY[parentNode] = 0.0f;
            octree.comZ[parentNode] = 0.0f;
        }

        completedNode = parentNode;
        parentNode = octree.parent[completedNode];
    }
}

void allocateOctreePhysicalData(Octree& octree)
{
    if (octree.nNodes <= 0)
        throw std::invalid_argument("Octree topology must be allocated first");

    if (!octree.mass.empty() || !octree.comX.empty() ||
        !octree.comY.empty() || !octree.comZ.empty() ||
        !octree.pendingChildren.empty() || !octree.childCount.empty())
        throw std::logic_error("Octree physical data is already allocated");

    const std::size_t count = static_cast<std::size_t>(octree.nNodes);
    thrust::device_vector<double> mass(count);
    thrust::device_vector<double> comX(count);
    thrust::device_vector<double> comY(count);
    thrust::device_vector<double> comZ(count);
    thrust::device_vector<int> pendingChildren(count);
    thrust::device_vector<int> childCount(count);
    octree.mass = std::move(mass);
    octree.comX = std::move(comX);
    octree.comY = std::move(comY);
    octree.comZ = std::move(comZ);
    octree.pendingChildren = std::move(pendingChildren);
    octree.childCount = std::move(childCount);
}

void computeOctreeMassAndCenterOfMass(Octree& octree, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ)
{
    if (octree.nParticles <= 0 || octree.nLeaves <= 0 || octree.nNodes <= 0 || octree.root != 0)
        throw std::logic_error("Octree topology is not valid");

    if (octree.children.empty() || octree.parent.empty() ||
        octree.firstParticle.empty() || octree.particleCount.empty())
        throw std::logic_error("Octree topology arrays are not allocated");

    if (octree.mass.empty() || octree.comX.empty() || octree.comY.empty() ||
        octree.comZ.empty() || octree.pendingChildren.empty() ||
        octree.childCount.empty())
        throw std::logic_error("Octree physical arrays are not allocated");

    if (d_sortedIndices == nullptr)
        throw std::invalid_argument("d_sortedIndices must not be null");

    if (d_mass == nullptr || d_positionX == nullptr || d_positionY == nullptr || d_positionZ == nullptr)
        throw std::invalid_argument("Particle arrays must not be null");

    const std::size_t nodeBytes = static_cast<std::size_t>(octree.nNodes) * sizeof(double);
    const OctreeDeviceView octreeView = octree.device_view();

    checkCuda(cudaMemset(octreeView.mass, 0, nodeBytes), "clear octree.mass");
    checkCuda(cudaMemset(octreeView.comX, 0, nodeBytes), "clear octree.comX");
    checkCuda(cudaMemset(octreeView.comY, 0, nodeBytes), "clear octree.comY");
    checkCuda(cudaMemset(octreeView.comZ, 0, nodeBytes), "clear octree.comZ");

    const std::size_t integerNodeBytes = static_cast<std::size_t>(octree.nNodes) * sizeof(int);

    checkCuda(cudaMemset(octreeView.pendingChildren, 0, integerNodeBytes), "clear octree.pendingChildren");
    checkCuda(cudaMemset(octreeView.childCount, 0, integerNodeBytes), "clear octree.childCount");
    
    thrust::device_vector<int> errorCodeDevice(1, 0);
    int* const errorCode = nbody::deviceData(errorCodeDevice);

    constexpr int threadsPerBlock = 256;

    const int blocks = (octree.nNodes + threadsPerBlock - 1) / threadsPerBlock;

    //determine how many children every node has.
    countOctreeChildrenKernel<<<blocks, threadsPerBlock>>>(octreeView, errorCode);
    checkCuda(cudaGetLastError(), "countOctreeChildrenKernel launch");
    checkCuda(cudaDeviceSynchronize(), "countOctreeChildrenKernel execution");
    throwIfDeviceError(errorCode);

    //initialize occupied leaves.
    initializeOctreeLeavesKernel<<<blocks, threadsPerBlock>>>(octreeView, d_sortedIndices, d_mass, d_positionX, d_positionY, d_positionZ, errorCode);
    checkCuda(cudaGetLastError(), "initializeOctreeLeavesKernel launch");
    checkCuda(cudaDeviceSynchronize(), "initializeOctreeLeavesKernel execution");
    throwIfDeviceError(errorCode);

    //bottom-up mass/CoM aggregation.
    propagateOctreeMassKernel<<<blocks, threadsPerBlock>>>(octreeView, errorCode);
    checkCuda(cudaGetLastError(), "propagateOctreeMassKernel launch");
    checkCuda(cudaDeviceSynchronize(), "propagateOctreeMassKernel execution");
    throwIfDeviceError(errorCode);
}
