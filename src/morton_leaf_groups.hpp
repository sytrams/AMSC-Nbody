#pragma once

#include <cstddef>
#include <cstdint>

struct MortonLeafGroups
{
    int nParticles = 0;
    int nGroups = 0;    //numero di differenti morton code
    int capacity = 0;   //max nr. particelle gestibili dagli array allocati
    std::uint32_t* uniqueKeys = nullptr;    //una chiave di morton per ogni run di chiavi uguali
    int* firstParticle = nullptr;   //posizione iniziale del gruppo nell'array morton ordinato
    int* particleCount = nullptr;   //nr. di particelle appartenenti a ogni gruppo
    int* numberOfGroupsDevice = nullptr;    //CUB scrive qui il numero di run trovati, è un singolo intero residente sulla GPU
    void* temporaryStorage = nullptr;   //buffer per CUB
    std::size_t temporaryStorageBytes = 0;
};

void allocateMortonLeafGroups(MortonLeafGroups& groups, int capacity);

void buildMortonLeafGroups(MortonLeafGroups& groups, const std::uint32_t* d_sortedKeys, int nParticles);

void freeMortonLeafGroups(MortonLeafGroups& groups);
