#pragma once

#include <cstddef>
#include <cstdint>

#include "device_buffer.hpp"

struct MortonLeafGroupsView
{
    int nParticles = 0;
    int nGroups = 0;    //numero di differenti morton code
    int capacity = 0;   //max nr. particelle gestibili dagli array allocati
    const std::uint32_t* uniqueKeys = nullptr;    //una chiave di morton per ogni run di chiavi uguali
    const int* firstParticle = nullptr;   //posizione iniziale del gruppo nell'array morton ordinato
    const int* particleCount = nullptr;   //nr. di particelle appartenenti a ogni gruppo  
};

struct MortonLeafGroups
{
    int nParticles = 0;
    int nGroups = 0;    
    int capacity = 0; 

    DeviceBuffer<std::uint32_t> uniqueKeys;
    DeviceBuffer<int> firstParticle;
    DeviceBuffer<int> particleCount;
    DeviceBuffer<int> numberOfGroupsDevice;
    DeviceBuffer<std::byte> temporaryStorage;

    [[nodiscard]]
    MortonLeafGroupsView view() const noexcept
    {
        return {
            nParticles,
            nGroups,
            capacity,
            uniqueKeys.data(),
            firstParticle.data(),
            particleCount.data()};
    }
};

void allocateMortonLeafGroups(MortonLeafGroups& groups, int capacity);

void buildMortonLeafGroups(MortonLeafGroups& groups, const std::uint32_t* d_sortedKeys, int nParticles);

void freeMortonLeafGroups(MortonLeafGroups& groups) noexcept;
