#pragma once

#include <cstddef>
#include <cstdint>

#include <thrust/device_vector.h>

using MortonKey = std::uint32_t;
using QuantisedCoordinate = std::uint32_t;
using ParticleIndex = std::uint32_t;

class Bbox;

class MortonKeys {
private:
    static constexpr std::size_t dimensions = 3;

    thrust::device_vector<MortonKey> keys_;
    thrust::device_vector<ParticleIndex> indices_;

    void build(const double* x, const double* y, const double* z, std::size_t count, const Bbox& boundingBox);

    void normalise(const double* x, const double* y, const double* z,
                   const double* centre_x, const double* centre_y,
                   const double* centre_z, const double* side, double* out_x,
                   double* out_y, double* out_z, std::size_t n);

    void quantise(const double* out_x, const double* out_y,
                  const double* out_z, QuantisedCoordinate* q_x,
                  QuantisedCoordinate* q_y, QuantisedCoordinate* q_z,
                  std::uint32_t grid_size, std::uint32_t max_q,
                  std::size_t n);

    void create_keys(const QuantisedCoordinate* q_x,
                     const QuantisedCoordinate* q_y,
                     const QuantisedCoordinate* q_z, std::size_t n);

public:
    // Coordinate pointers must each reference at least count doubles in device
    // memory. Keys and particle indices are stored in ascending Morton order.
    MortonKeys(const double* x, const double* y, const double* z, std::size_t count);
    MortonKeys(const double* x, const double* y, const double* z, std::size_t count, const Bbox& boundingBox);

    [[nodiscard]] const MortonKey *keys_device_data() const noexcept;
    [[nodiscard]] MortonKey *keys_device_data() noexcept;
    [[nodiscard]] const ParticleIndex *indices_device_data() const noexcept;
    [[nodiscard]] ParticleIndex *indices_device_data() noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
};
