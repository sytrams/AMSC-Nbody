#include "particle.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace
{
struct ParticleFileHeader
{
    std::uint32_t num_particles;
    std::streamoff header_bytes;
};

ParticleFileHeader detectParticleFileHeader(std::ifstream& input)
{
    input.clear();
    input.seekg(0, std::ios::end);
    const auto fileSize = input.tellg();
    if (fileSize < 0)
    {
        throw std::runtime_error("Failed to determine particle file size.");
    }

    constexpr std::streamoff bytesPerParticle =
        static_cast<std::streamoff>(7 * sizeof(double));

    input.seekg(0, std::ios::beg);
    std::uint64_t count64 = 0;
    input.read(reinterpret_cast<char*>(&count64), sizeof(count64));
    if (input &&
        count64 > 0 &&
        count64 <= static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max()) &&
        fileSize == static_cast<std::streamoff>(sizeof(std::uint64_t)) +
                        static_cast<std::streamoff>(count64) * bytesPerParticle)
    {
        return {
            static_cast<std::uint32_t>(count64),
            static_cast<std::streamoff>(sizeof(std::uint64_t))};
    }

    input.clear();
    input.seekg(0, std::ios::beg);
    std::uint32_t count32 = 0;
    input.read(reinterpret_cast<char*>(&count32), sizeof(count32));
    if (input &&
        count32 > 0 &&
        fileSize == static_cast<std::streamoff>(sizeof(std::uint32_t)) +
                        static_cast<std::streamoff>(count32) * bytesPerParticle)
    {
        return {
            count32,
            static_cast<std::streamoff>(sizeof(std::uint32_t))};
    }

    throw std::runtime_error(
        "Unsupported particle file layout. Expected planar doubles with a "
        "32-bit or 64-bit particle count header.");
}
} // namespace

Particles::Particles(std::ifstream& input)
{
    if (!input.is_open() || !input.good())
    {
        throw std::runtime_error("Invalid or unreadable file stream provided.");
    }

    const auto header = detectParticleFileHeader(input);
    num_particles = header.num_particles;

    mass = std::make_unique<double[]>(num_particles);
    x = std::make_unique<double[]>(num_particles);
    y = std::make_unique<double[]>(num_particles);
    z = std::make_unique<double[]>(num_particles);
    vx = std::make_unique<double[]>(num_particles);
    vy = std::make_unique<double[]>(num_particles);
    vz = std::make_unique<double[]>(num_particles);

    input.clear();
    input.seekg(header.header_bytes, std::ios::beg);

    const auto bytesToRead =
        static_cast<std::streamsize>(num_particles * sizeof(double));
    input.read(reinterpret_cast<char*>(mass.get()), bytesToRead);
    input.read(reinterpret_cast<char*>(x.get()), bytesToRead);
    input.read(reinterpret_cast<char*>(y.get()), bytesToRead);
    input.read(reinterpret_cast<char*>(z.get()), bytesToRead);
    input.read(reinterpret_cast<char*>(vx.get()), bytesToRead);
    input.read(reinterpret_cast<char*>(vy.get()), bytesToRead);
    input.read(reinterpret_cast<char*>(vz.get()), bytesToRead);

    if (!input)
    {
        throw std::runtime_error(
            "Failed to read all expected particle data. File may be truncated.");
    }
}
