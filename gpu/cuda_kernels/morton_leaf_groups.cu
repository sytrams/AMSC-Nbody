#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "morton_leaf_groups.hpp"

namespace
{
    void checkCuda (cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + "failed: " + cudaGetErrorString(error));
    }

    template <typename T>
    void allocateDeviceArray(T*& pointer, std::size_t count, const char* name)
    {
        if (count == 0)
            return;
        
        const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T));
        
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("cudaMalloc ") + name + " failed: " + cudaGetErrorString(error));
    }

    void ensureTemporaryStorage(MortonLeafGroups& groups, std::size_t requiredBytes)
    {
        if (requiredBytes <= groups.temporaryStorageBytes)
            return;

        void* newStorage = nullptr;

        checkCuda(cudaMalloc(&newStorage, requiredBytes), "cudaMalloc MortonLeafGroups temporary storage");

        // Libera il buffer vecchio solo dopo che la nuova allocazione è riuscita.
        cudaFree(groups.temporaryStorage);

        groups.temporaryStorage = newStorage;
        groups.temporaryStorageBytes = requiredBytes;
    }
}//namespace

void allocateMortonLeafGroups(MortonLeafGroups& groups, int capacity)
{
    if (capacity <= 0)
        throw std::invalid_argument("MortonLeafGroups capacity must be positive");

    if (groups.uniqueKeys != nullptr || groups.firstParticle != nullptr || groups.particleCount != nullptr || groups.numberOfGroupsDevice != nullptr || groups.temporaryStorage != nullptr)
        throw std::logic_error("MortonLeafGroups memory is already allocated");

    const std::size_t arraySize = static_cast<std::size_t>(capacity);

    try
    {
        allocateDeviceArray(groups.uniqueKeys, arraySize, "groups.uniqueKeys");
        allocateDeviceArray(groups.firstParticle, arraySize,"groups.firstParticle");
        allocateDeviceArray(groups.particleCount, arraySize, "groups.particleCount");
        allocateDeviceArray(groups.numberOfGroupsDevice, 1, "groups.numberOfGroupsDevice");
    }
    catch(...)
    {
        freeMortonLeafGroups(groups);
        throw;
    }
    
    groups.capacity = capacity;
    groups.nParticles = 0;
    groups.nGroups = 0;
}

void buildMortonLeafGroups(MortonLeafGroups& groups, const std::uint32_t* d_sortedKeys, int nParticles)
{
    if (nParticles <= 0)
        throw std::invalid_argument("nParticles exceeds MortonLeafGroups capacity");
    
    if (d_sortedKeys == nullptr)
        throw std::invalid_argument("d_sortedKeys must not be null");

    if (groups.uniqueKeys == nullptr || groups.firstParticle == nullptr || groups.particleCount == nullptr || groups.numberOfGroupsDevice == nullptr)
       throw std::logic_error("MortonLeafGroups memory has not been allocated");
       
    //Evita che il chiamante possa leggere metadati vecchi nel caso una delle operazioni seguenti fallisca
    groups.nParticles = 0;
    groups.nGroups = 0;

    //Determinazione della memoria temporanea richiesta da DeviceRunLengthEncode
    std::size_t runLengthStorageBytes = 0;

    checkCuda(cub::DeviceRunLengthEncode::Encode(nullptr, runLengthStorageBytes, d_sortedKeys, groups.uniqueKeys, groups.particleCount, groups.numberOfGroupsDevice, nParticles), "DeviceRunLengthEncode storage query");
    ensureTemporaryStorage(groups, runLengthStorageBytes);

    checkCuda(cub::DeviceRunLengthEncode::Encode(groups.temporaryStorage, runLengthStorageBytes, d_sortedKeys, groups.uniqueKeys, groups.particleCount, groups.numberOfGroupsDevice,nParticles), "DeviceRunLengthEncode execution");
    int numberOfGroups = 0;

    //cudaMemcpy DeviceToHost attende che l'operazione precedente sul default stream sia terminata
    checkCuda(cudaMemcpy(&numberOfGroups, groups.numberOfGroupsDevice, sizeof(int), cudaMemcpyDeviceToHost), "copy number of Morton groups to host");
    if (numberOfGroups <= 0 || numberOfGroups > nParticles)
        throw std::runtime_error("DeviceRunLengthEncode produced an invalid group count");
    
    //Determinzione della memoria richiesta dalla exclusive prefix sum
    std::size_t scanStorageBytes = 0;

    checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, scanStorageBytes, groups.particleCount, groups.firstParticle, numberOfGroups), "DeviceScan storage query");

    ensureTemporaryStorage(groups, scanStorageBytes);

    //Exclusive prefix sum
    checkCuda(cub::DeviceScan::ExclusiveSum(groups.temporaryStorage, scanStorageBytes, groups.particleCount, groups.firstParticle, numberOfGroups), "DeviceScan execution");

    checkCuda(cudaDeviceSynchronize(), "buildMortonLeafGroups synchronization");
    groups.nParticles = nParticles;
    groups.nGroups = numberOfGroups;
}

void freeMortonLeafGroups(MortonLeafGroups& groups)
{
    cudaFree(groups.uniqueKeys);
    cudaFree(groups.firstParticle);
    cudaFree(groups.particleCount);
    cudaFree(groups.numberOfGroupsDevice);
    cudaFree(groups.temporaryStorage);

    groups.uniqueKeys = nullptr;
    groups.firstParticle = nullptr;
    groups.particleCount = nullptr;
    groups.numberOfGroupsDevice = nullptr;
    groups.temporaryStorage = nullptr;

    groups.nParticles = 0;
    groups.nGroups = 0;
    groups.capacity = 0;
    groups.temporaryStorageBytes = 0;
}