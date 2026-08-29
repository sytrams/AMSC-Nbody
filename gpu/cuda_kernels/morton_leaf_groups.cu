#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "morton_leaf_groups.hpp"

namespace
{
    void checkCuda (cudaError_t error, const char* operation)
    {
        if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }

    void ensureTemporaryStorage(MortonLeafGroups& groups, std::size_t requiredBytes)
    {
        if (requiredBytes <= groups.temporaryStorage.size())
            return;

        DeviceBuffer<std::byte> newStorage(requiredBytes);
        groups.temporaryStorage = std::move(newStorage);
    }
}//namespace

void allocateMortonLeafGroups(MortonLeafGroups& groups, int capacity)
{
    if (capacity <= 0)
        throw std::invalid_argument("MortonLeafGroups capacity must be positive");

    if (!groups.uniqueKeys.empty() || !groups.firstParticle.empty() || !groups.particleCount.empty() || !groups.numberOfGroupsDevice.empty() || !groups.temporaryStorage.empty())
        throw std::logic_error("MortonLeafGroups memory is already allocated");

    const std::size_t arraySize = static_cast<std::size_t>(capacity);

    try
    {
        groups.uniqueKeys.allocate(arraySize);
        groups.firstParticle.allocate(arraySize);
        groups.particleCount.allocate(arraySize);
        groups.numberOfGroupsDevice.allocate(1);
    }
    catch (...)
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
        throw std::invalid_argument("buildMortonLeafGroups requires nParticles >= 1");
    
    if (nParticles > groups.capacity)
        throw std::invalid_argument("nParticles exceeds MortonLeafGroups capacity");

    if (d_sortedKeys == nullptr)
        throw std::invalid_argument("d_sortedKeys must not be null");

    if (groups.uniqueKeys.data() == nullptr || groups.firstParticle.data() == nullptr || groups.particleCount.data() == nullptr || groups.numberOfGroupsDevice.data() == nullptr)
       throw std::logic_error("MortonLeafGroups memory has not been allocated");
       
    //Evita che il chiamante possa leggere metadati vecchi nel caso una delle operazioni seguenti fallisca
    groups.nParticles = 0;
    groups.nGroups = 0;

    //Determinazione della memoria temporanea richiesta da DeviceRunLengthEncode
    std::size_t runLengthStorageBytes = 0;

    checkCuda(cub::DeviceRunLengthEncode::Encode(nullptr, runLengthStorageBytes, d_sortedKeys, groups.uniqueKeys.data(), groups.particleCount.data(), groups.numberOfGroupsDevice.data(), nParticles), "DeviceRunLengthEncode storage query");
    ensureTemporaryStorage(groups, runLengthStorageBytes);

    checkCuda(cub::DeviceRunLengthEncode::Encode(groups.temporaryStorage.data(), runLengthStorageBytes, d_sortedKeys, groups.uniqueKeys.data(), groups.particleCount.data(), groups.numberOfGroupsDevice.data(),nParticles), "DeviceRunLengthEncode execution");
    int numberOfGroups = 0;

    //cudaMemcpy DeviceToHost attende che l'operazione precedente sul default stream sia terminata
    checkCuda(cudaMemcpy(&numberOfGroups, groups.numberOfGroupsDevice.data(), sizeof(int), cudaMemcpyDeviceToHost), "copy number of Morton groups to host");
    if (numberOfGroups <= 0 || numberOfGroups > nParticles)
        throw std::runtime_error("DeviceRunLengthEncode produced an invalid group count");
    
    //Determinzione della memoria richiesta dalla exclusive prefix sum
    std::size_t scanStorageBytes = 0;

    checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, scanStorageBytes, groups.particleCount.data(), groups.firstParticle.data(), numberOfGroups), "DeviceScan storage query");

    ensureTemporaryStorage(groups, scanStorageBytes);

    //Exclusive prefix sum
    checkCuda(cub::DeviceScan::ExclusiveSum(groups.temporaryStorage.data(), scanStorageBytes, groups.particleCount.data(), groups.firstParticle.data(), numberOfGroups), "DeviceScan execution");

    checkCuda(cudaDeviceSynchronize(), "buildMortonLeafGroups synchronization");
    groups.nParticles = nParticles;
    groups.nGroups = numberOfGroups;
}
void freeMortonLeafGroups(MortonLeafGroups& groups) noexcept
{
    groups = MortonLeafGroups{};
}