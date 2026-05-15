module;

#include <string>
#include <iostream>
#include <fstream>

export module particle;

export inline constexpr double G = 6.67430e-11;  // gravitational constant
export inline constexpr double K = 8.98755e9;   // Coulomb constant

namespace {
struct ParticleFileHeader {
    uint32_t num_particles;
    std::streamoff header_bytes;
};

ParticleFileHeader detect_particle_file_header(std::ifstream& inFile) {
    inFile.clear();
    inFile.seekg(0, std::ios::end);
    const auto file_size = inFile.tellg();
    if (file_size < 0) {
        throw std::runtime_error("Failed to determine particle file size.");
    }

    constexpr std::streamoff bytes_per_particle = static_cast<std::streamoff>(6 * sizeof(double));

    inFile.seekg(0, std::ios::beg);
    uint64_t count64 = 0;
    inFile.read(reinterpret_cast<char*>(&count64), sizeof(count64));
    if (inFile
        && count64 > 0
        && count64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        && file_size == static_cast<std::streamoff>(sizeof(uint64_t)) + static_cast<std::streamoff>(count64) * bytes_per_particle) {
        return {static_cast<uint32_t>(count64), static_cast<std::streamoff>(sizeof(uint64_t))};
    }

    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    uint32_t count32 = 0;
    inFile.read(reinterpret_cast<char*>(&count32), sizeof(count32));
    if (inFile
        && count32 > 0
        && file_size == static_cast<std::streamoff>(sizeof(uint32_t)) + static_cast<std::streamoff>(count32) * bytes_per_particle) {
        return {count32, static_cast<std::streamoff>(sizeof(uint32_t))};
    }

    throw std::runtime_error("Unsupported particle file layout. Expected planar doubles with a 32-bit or 64-bit particle count header.");
}
}  // namespace

export class Particles {
public:
    uint32_t num_particles;
    std::unique_ptr<double[]> x, y, z;
    std::unique_ptr<double[]> vx, vy, vz;

    explicit Particles(std::ifstream& inFile) {
        if (!inFile.is_open() || !inFile.good()) {
            throw std::runtime_error("Invalid or unreadable file stream provided.");
        }

        const auto header = detect_particle_file_header(inFile);
        num_particles = header.num_particles;
        if (num_particles == 0) {
            throw std::runtime_error("Particle count is zero.");
        }

        x = std::make_unique<double[]>(num_particles);
        y = std::make_unique<double[]>(num_particles);
        z = std::make_unique<double[]>(num_particles);
        vx = std::make_unique<double[]>(num_particles);
        vy = std::make_unique<double[]>(num_particles);
        vz = std::make_unique<double[]>(num_particles);

        inFile.clear();
        inFile.seekg(header.header_bytes, std::ios::beg);

        size_t bytes_to_read = num_particles * sizeof(double);
        inFile.read(reinterpret_cast<char*>(x.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(y.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(z.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(vx.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(vy.get()), bytes_to_read);
        inFile.read(reinterpret_cast<char*>(vz.get()), bytes_to_read);

        if (!inFile) {
            throw std::runtime_error("Failed to read all expected particle data. File may be truncated.");
        }
    }
};
