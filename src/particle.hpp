#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

inline constexpr double G = 6.67430e-11; // gravitational constant
inline constexpr double K = 8.98755e9;   // Coulomb constant

namespace {
struct ParticleFileHeader {
  uint32_t num_particles;
  std::streamoff header_bytes;
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

  inFile.seekg(0, std::ios::beg);
  uint64_t count64 = 0;
  inFile.read(reinterpret_cast<char *>(&count64), sizeof(count64));
  if (inFile && count64 > 0 &&
      count64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) &&
      file_size ==
          static_cast<std::streamoff>(sizeof(uint64_t)) +
              static_cast<std::streamoff>(count64) * bytes_per_particle) {
    return {static_cast<uint32_t>(count64),
            static_cast<std::streamoff>(sizeof(uint64_t))};
  }

  inFile.clear();
  inFile.seekg(0, std::ios::beg);
  uint32_t count32 = 0;
  inFile.read(reinterpret_cast<char *>(&count32), sizeof(count32));
  if (inFile && count32 > 0 &&
      file_size ==
          static_cast<std::streamoff>(sizeof(uint32_t)) +
              static_cast<std::streamoff>(count32) * bytes_per_particle) {
    return {count32, static_cast<std::streamoff>(sizeof(uint32_t))};
  }

  throw std::runtime_error(
      "Unsupported particle file layout. Expected planar doubles with a 32-bit "
      "or 64-bit particle count header.");
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
};

class Particles {
private:
  // Device Data
  uint32_t num_particles_;
  thrust::device_vector<double> mass_;
  thrust::device_vector<double> x_, y_, z_;
  thrust::device_vector<double> vx_, vy_, vz_;

  // Pointers to Data on GPU
  DeviceParticlesView particles_env_{};

public:
  DeviceParticlesView device_view() {
    return {num_particles_,
            thrust::raw_pointer_cast(mass_.data()),
            thrust::raw_pointer_cast(x_.data()),
            thrust::raw_pointer_cast(y_.data()),
            thrust::raw_pointer_cast(z_.data()),
            thrust::raw_pointer_cast(vx_.data()),
            thrust::raw_pointer_cast(vy_.data()),
            thrust::raw_pointer_cast(vz_.data())};
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

    inFile.clear();
    inFile.seekg(header.header_bytes, std::ios::beg);

    std::vector<double> host_buffer(num_particles_);

    const auto bytes =
        static_cast<std::streamsize>(num_particles_ * sizeof(double));

    auto read_block = [&](thrust::device_vector<double> &destination,
                          const std::string &block_name) {
      inFile.read(reinterpret_cast<char *>(host_buffer.data()), bytes);

      if (inFile.gcount() != bytes) {
        throw std::runtime_error("Incomplete " + block_name + " block");
      }

      thrust::copy(host_buffer.begin(), host_buffer.end(), destination.begin());
    };

    read_block(mass_, "mass");

    read_block(x_, "x position");
    read_block(y_, "y position");
    read_block(z_, "z position");

    read_block(vx_, "x velocity");
    read_block(vy_, "y velocity");
    read_block(vz_, "z velocity");

    if (!inFile) {
      throw std::runtime_error(
          "Failed to read all expected particle data. File may be truncated.");
    }

    this->particles_env_ = device_view();
  }

  std::size_t size() const { return num_particles_; }
};
