#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "particle_type.hpp"
#include "streaming/frame.hpp"

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("Expected one output directory argument");
    const std::filesystem::path directory(argv[1]);
    nbody::streaming::FrameSpool spool(directory, "browser-test");
    const std::array<std::uint8_t, 4> types{
        static_cast<std::uint8_t>(ParticleType::Star),
        static_cast<std::uint8_t>(ParticleType::Planet),
        static_cast<std::uint8_t>(ParticleType::Moon),
        static_cast<std::uint8_t>(ParticleType::Asteroid)};
    const nbody::streaming::PositionBounds bounds{{-2.0, -2.0, -1.0},
                                                  {2.0, 2.0, 1.0}};

    for (std::uint64_t sequence = 0; sequence < 2; ++sequence) {
      (void)spool.writeQuantizedPositions(
          sequence, sequence * 10, static_cast<double>(sequence) * 0.5,
          types.size(), bounds,
          [sequence](std::size_t offset, std::size_t count,
                     std::uint16_t *positions) {
            for (std::size_t index = 0; index < count; ++index) {
              const std::size_t particle = offset + index;
              positions[index * 3] = static_cast<std::uint16_t>(
                  8000 + particle * 14000 + sequence * 1000);
              positions[index * 3 + 1] = static_cast<std::uint16_t>(
                  56000 - particle * 12000 - sequence * 1000);
              positions[index * 3 + 2] = static_cast<std::uint16_t>(
                  16000 + particle * 9000);
            }
          },
          types.size(),
          [&types](std::size_t offset, std::size_t count,
                   std::uint8_t *output) {
            std::copy_n(types.data() + offset, count, output);
          });
    }
    (void)spool.markComplete();
    std::cout << directory << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
