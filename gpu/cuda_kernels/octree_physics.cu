#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "octree_physics.hpp"

namespace 
{
    void checkCuda(cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }

    template <typename T>
    void allocateDeviceArray(T*& pointer, std::size_t count, const char* name)
    {
        if (count == 0)
            return;

        const cudaError_t error =cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T));

        if (error != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc ") + name + " failed: " + cudaGetErrorString(error));
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

__global__ void countOctreeChildrenKernel(Octree octree, int* errorCode)
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

__global__ void initializeOctreeLeavesKernel(Octree octree, const std::uint32_t* sortedIndices, const double* particleMass, const double* positionX, const double* positionY, const double* positionZ, int* errorCode)
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

__global__ void propagateOctreeMassKernel(Octree octree, int* errorCode)
{
    const int leaf = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (leaf >= octree.nNodes)
        return;

    //Only occupied level-10 cells start a bottom-up propagation.
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

    if (octree.mass != nullptr || octree.comX != nullptr || octree.comY != nullptr || octree.comZ != nullptr || octree.pendingChildren != nullptr || octree.childCount != nullptr)
        throw std::logic_error("Octree physical data is already allocated");

    const std::size_t count = static_cast<std::size_t>(octree.nNodes);

    try
    {
        allocateDeviceArray(octree.mass, count, "octree.mass");
        allocateDeviceArray(octree.comX, count, "octree.comX");
        allocateDeviceArray(octree.comY, count, "octree.comY");
        allocateDeviceArray(octree.comZ, count, "octree.comZ");
        allocateDeviceArray(octree.pendingChildren, count, "octree.pendingChildren");
        allocateDeviceArray(octree.childCount, count, "octree.childCount");
    }
    catch (...)
    {
        //freeOctree() already frees these arrays,but it also frees topology. Therefore release only what was allocated here if this function fails.
        cudaFree(octree.mass);
        cudaFree(octree.comX);
        cudaFree(octree.comY);
        cudaFree(octree.comZ);
        cudaFree(octree.pendingChildren);
        cudaFree(octree.childCount);

        octree.mass = nullptr;
        octree.comX = nullptr;
        octree.comY = nullptr;
        octree.comZ = nullptr;
        octree.pendingChildren = nullptr;
        octree.childCount = nullptr;

        throw;
    }
}

void computeOctreeMassAndCenterOfMass(Octree& octree, const std::uint32_t* d_sortedIndices, const double* d_mass, const double* d_positionX, const double* d_positionY, const double* d_positionZ)
{
    if (octree.nParticles <= 0 || octree.nLeaves <= 0 || octree.nNodes <= 0 || octree.root != 0)
        throw std::logic_error("Octree topology is not valid");

    if (octree.children == nullptr || octree.parent == nullptr || octree.firstParticle == nullptr || octree.particleCount == nullptr)
        throw std::logic_error("Octree topology arrays are not allocated");

    if (octree.mass == nullptr || octree.comX == nullptr || octree.comY == nullptr || octree.comZ == nullptr || octree.pendingChildren == nullptr || octree.childCount == nullptr)
        throw std::logic_error("Octree physical arrays are not allocated");

    if (d_sortedIndices == nullptr)
        throw std::invalid_argument("d_sortedIndices must not be null");

    if (d_mass == nullptr || d_positionX == nullptr || d_positionY == nullptr || d_positionZ == nullptr)
        throw std::invalid_argument("Particle arrays must not be null");

    const std::size_t nodeBytes = static_cast<std::size_t>(octree.nNodes) * sizeof(double);

    checkCuda(cudaMemset(octree.mass, 0, nodeBytes), "clear octree.mass");
    checkCuda(cudaMemset(octree.comX, 0, nodeBytes), "clear octree.comX");
    checkCuda(cudaMemset( octree.comY, 0, nodeBytes), "clear octree.comY");
    checkCuda(cudaMemset(octree.comZ, 0, nodeBytes), "clear octree.comZ");

    const std::size_t integerNodeBytes = static_cast<std::size_t>(octree.nNodes) * sizeof(int);

    checkCuda(cudaMemset(octree.pendingChildren, 0, integerNodeBytes), "clear octree.pendingChildren");
    checkCuda(cudaMemset(octree.childCount, 0, integerNodeBytes), "clear octree.childCount");
    
    int* errorCodeDevice = nullptr;

    try
    {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&errorCodeDevice), sizeof(int)), "allocate octree physics error code");
        checkCuda(cudaMemset(errorCodeDevice, 0, sizeof(int)), "clear octree physics error code");

        constexpr int threadsPerBlock = 256;

        const int blocks = (octree.nNodes + threadsPerBlock - 1) / threadsPerBlock;

        //determine how many children every node has.
        countOctreeChildrenKernel<<<blocks, threadsPerBlock>>>(octree, errorCodeDevice);
        checkCuda(cudaGetLastError(), "countOctreeChildrenKernel launch");
        checkCuda(cudaDeviceSynchronize(), "countOctreeChildrenKernel execution");
        throwIfDeviceError(errorCodeDevice);

        //initialize occupied leaves.
        
        initializeOctreeLeavesKernel<<<blocks, threadsPerBlock>>>( octree, d_sortedIndices, d_mass, d_positionX, d_positionY, d_positionZ, errorCodeDevice);
        checkCuda(cudaGetLastError(), "initializeOctreeLeavesKernel launch");
        checkCuda(cudaDeviceSynchronize(), "initializeOctreeLeavesKernel execution");
        throwIfDeviceError(errorCodeDevice);

        //bottom-up mass/CoM aggregation.
        propagateOctreeMassKernel<<<blocks, threadsPerBlock>>>(octree, errorCodeDevice);
        checkCuda(cudaGetLastError(), "propagateOctreeMassKernel launch");
        checkCuda(cudaDeviceSynchronize(), "propagateOctreeMassKernel execution");
        throwIfDeviceError(errorCodeDevice);
        cudaFree(errorCodeDevice);
        errorCodeDevice = nullptr;
    }
    catch (...)
    {
        cudaFree(errorCodeDevice);
        throw;
    }
}