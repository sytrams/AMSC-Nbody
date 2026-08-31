#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include "particle_type.hpp"

inline constexpr double G = 6.67430e-11; // gravitational constant
inline constexpr double K = 8.98755e9;   // Coulomb constant

namespace {
struct ParticleFileHeader {
  uint32_t num_particles;
  std::streamoff header_bytes;
  bool has_particle_types;
};

ParticleFileHeader detect_particle_file_header(std::ifstream &inFile) {
  inFile.clear();
  inFile.seekg(0, std::ios::end);
  const auto file_size = inFile.tellg();
  if (file_size < 0) {
    throw std::runtime_error("Failed to determine particle file size.");
  }

  constexpr std::streamoff bytes_per_particle =
      static_cast<std::streamoff>(7 * sizeof(double));

  const auto matches_layout = [&](std::uint64_t count,
                                  std::streamoff header_bytes,
                                  bool has_particle_types) {
    const auto type_bytes = has_particle_types
                                ? static_cast<std::streamoff>(count)
                                : std::streamoff{0};
    return file_size == header_bytes +
                            static_cast<std::streamoff>(count) *
                                bytes_per_particle +
                            type_bytes;
  };

  inFile.seekg(0, std::ios::beg);
  uint64_t count64 = 0;
  inFile.read(reinterpret_cast<char *>(&count64), sizeof(count64));
  if (inFile && count64 > 0 &&
      count64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    const auto header_bytes = static_cast<std::streamoff>(sizeof(uint64_t));
    if (matches_layout(count64, header_bytes, true)) {
      return {static_cast<uint32_t>(count64), header_bytes, true};
    }
    if (matches_layout(count64, header_bytes, false)) {
      return {static_cast<uint32_t>(count64), header_bytes, false};
    }
  }

  inFile.clear();
  inFile.seekg(0, std::ios::beg);
  uint32_t count32 = 0;
  inFile.read(reinterpret_cast<char *>(&count32), sizeof(count32));
  if (inFile && count32 > 0) {
    const auto header_bytes = static_cast<std::streamoff>(sizeof(uint32_t));
    if (matches_layout(count32, header_bytes, true)) {
      return {count32, header_bytes, true};
    }
    if (matches_layout(count32, header_bytes, false)) {
      return {count32, header_bytes, false};
    }
  }

  throw std::runtime_error(
      "Unsupported particle file layout. Expected planar doubles with a 32-bit "
      "or 64-bit particle count header, optionally followed by one uint8 "
      "particle type per particle.");
}

} // namespace

struct DeviceParticlesView {
  std::size_t count;

  double *mass;
  double *x;
  double *y;
  double *z;
  double *vx;
  double *vy;
  double *vz;
  std::uint8_t *type;
};

struct ConstDeviceParticlesView {
  std::size_t count;

  const double *mass;
  const double *x;
  const double *y;
  const double *z;
  const double *vx;
  const double *vy;
  const double *vz;
  const std::uint8_t *type;
};

class Particles {
private:
  // Device Data
  uint32_t num_particles_;
  thrust::device_vector<double> mass_;
  thrust::device_vector<double> x_, y_, z_;
  thrust::device_vector<double> vx_, vy_, vz_;
  thrust::device_vector<std::uint8_t> type_;

public:
  [[nodiscard]] DeviceParticlesView device_view() noexcept {
    return {num_particles_,
            thrust::raw_pointer_cast(mass_.data()),
            thrust::raw_pointer_cast(x_.data()),
            thrust::raw_pointer_cast(y_.data()),
            thrust::raw_pointer_cast(z_.data()),
            thrust::raw_pointer_cast(vx_.data()),
            thrust::raw_pointer_cast(vy_.data()),
            thrust::raw_pointer_cast(vz_.data()),
            thrust::raw_pointer_cast(type_.data())};
  }

  [[nodiscard]] ConstDeviceParticlesView device_view() const noexcept {
    return {num_particles_,
            thrust::raw_pointer_cast(mass_.data()),
            thrust::raw_pointer_cast(x_.data()),
            thrust::raw_pointer_cast(y_.data()),
            thrust::raw_pointer_cast(z_.data()),
            thrust::raw_pointer_cast(vx_.data()),
            thrust::raw_pointer_cast(vy_.data()),
            thrust::raw_pointer_cast(vz_.data()),
            thrust::raw_pointer_cast(type_.data())};
  }

  explicit Particles(std::ifstream &inFile) {
    if (!inFile.is_open() || !inFile.good()) {
      throw std::runtime_error("Invalid or unreadable file stream provided.");
    }

    const auto header = detect_particle_file_header(inFile);
    num_particles_ = header.num_particles;
    if (num_particles_ == 0) {
      throw std::runtime_error("Particle count is zero.");
    }

    this->mass_.resize(num_particles_);
    this->x_.resize(num_particles_);
    this->y_.resize(num_particles_);
    this->z_.resize(num_particles_);
    this->vx_.resize(num_particles_);
    this->vy_.resize(num_particles_);
    this->vz_.resize(num_particles_);
    this->type_.resize(num_particles_);

    inFile.clear();
    inFile.seekg(header.header_bytes, std::ios::beg);

    std::vector<double> host_buffer(num_particles_);

    const auto bytes =
        static_cast<std::streamsize>(num_particles_ * sizeof(double));

    auto read_block = [&](thrust::device_vector<double> &destination,
                      const std::string &block_name,
                      bool require_non_negative = false) {
    inFile.read(reinterpret_cast<char *>(host_buffer.data()), bytes);

    if (inFile.gcount() != bytes) {
      throw std::runtime_error("Incomplete " + block_name + " block");
    }

    for (std::size_t index = 0; index < host_buffer.size(); ++index) {
      const double value = host_buffer[index];

      if (!std::isfinite(value)) 
        throw std::runtime_error("Non-finite " + block_name + " value at particle " + std::to_string(index));
      
      if (require_non_negative && value < 0.0) 
        throw std::runtime_error("Negative " + block_name +" value at particle " + std::to_string(index));
    }

    thrust::copy(host_buffer.begin(), host_buffer.end(), destination.begin());
  };

    read_block(mass_, "mass", true);

    read_block(x_, "x position");
    read_block(y_, "y position");
    read_block(z_, "z position");

    read_block(vx_, "x velocity");
    read_block(vy_, "y velocity");
    read_block(vz_, "z velocity");

    std::vector<std::uint8_t> host_types(
        num_particles_, static_cast<std::uint8_t>(ParticleType::Unknown));
    if (header.has_particle_types) {
      const auto type_bytes = static_cast<std::streamsize>(num_particles_);
      inFile.read(reinterpret_cast<char *>(host_types.data()), type_bytes);
      if (inFile.gcount() != type_bytes) {
        throw std::runtime_error("Incomplete particle type block");
      }
      for (const std::uint8_t type : host_types) {
        if (!isValidParticleType(type)) {
          throw std::runtime_error("Unsupported particle type value");
        }
      }
    }
    thrust::copy(host_types.begin(), host_types.end(), type_.begin());

    if (!inFile) {
      throw std::runtime_error(
          "Failed to read all expected particle data. File may be truncated.");
    }
  }
};
