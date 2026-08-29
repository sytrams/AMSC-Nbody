#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <thrust/device_vector.h>

#include "device_vector_utils.hpp"

// Trivially-copyable, non-owning kernel argument. MortonLeafGroups owns every
// allocation.
template <typename Uint, typename Int, typename Byte>
struct BasicMortonLeafGroupsDeviceView {
  int nParticles = 0;
  int nGroups = 0;
  int capacity = 0;
  Uint *uniqueKeys = nullptr;
  Int *firstParticle = nullptr;
  Int *particleCount = nullptr;
  Int *numberOfGroups = nullptr;
  Byte *temporaryStorage = nullptr;
  std::size_t temporaryStorageBytes = 0;
};

using MortonLeafGroupsDeviceView =
    BasicMortonLeafGroupsDeviceView<std::uint32_t, int, unsigned char>;
using ConstMortonLeafGroupsDeviceView =
    BasicMortonLeafGroupsDeviceView<const std::uint32_t, const int,
                                    const unsigned char>;
static_assert(std::is_trivially_copyable_v<MortonLeafGroupsDeviceView>);
static_assert(std::is_trivially_copyable_v<ConstMortonLeafGroupsDeviceView>);

struct MortonLeafGroups {
  int nParticles = 0;
  int nGroups = 0;  // numero di differenti morton code
  int capacity = 0; // max nr. particelle gestibili dagli array allocati
  thrust::device_vector<std::uint32_t>
      uniqueKeys; // una chiave di morton per ogni run di chiavi uguali
  thrust::device_vector<int>
      firstParticle; // posizione iniziale del gruppo nell'array morton ordinato
  thrust::device_vector<int>
      particleCount; // nr. di particelle appartenenti a ogni gruppo
  thrust::device_vector<int>
      numberOfGroupsDevice; // CUB scrive qui il numero di run trovati, è un
                            // singolo intero residente sulla GPU
  thrust::device_vector<unsigned char> temporaryStorage; // buffer per CUB

  [[nodiscard]] MortonLeafGroupsDeviceView device_view() noexcept {
    return {nParticles,
            nGroups,
            capacity,
            nbody::deviceData(uniqueKeys),
            nbody::deviceData(firstParticle),
            nbody::deviceData(particleCount),
            nbody::deviceData(numberOfGroupsDevice),
            nbody::deviceData(temporaryStorage),
            temporaryStorage.size()};
  }

  [[nodiscard]] ConstMortonLeafGroupsDeviceView device_view() const noexcept {
    return {nParticles,
            nGroups,
            capacity,
            nbody::deviceData(uniqueKeys),
            nbody::deviceData(firstParticle),
            nbody::deviceData(particleCount),
            nbody::deviceData(numberOfGroupsDevice),
            nbody::deviceData(temporaryStorage),
            temporaryStorage.size()};
  }
};

void allocateMortonLeafGroups(MortonLeafGroups &groups, int capacity);

void buildMortonLeafGroups(MortonLeafGroups &groups,
                           const std::uint32_t *d_sortedKeys, int nParticles);

void freeMortonLeafGroups(MortonLeafGroups &groups);
