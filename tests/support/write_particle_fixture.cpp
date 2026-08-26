#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " OUTPUT_FILE\n";
    return 2;
  }

  std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    std::cerr << "Could not create particle fixture: " << argv[1] << '\n';
    return 1;
  }

  constexpr std::uint64_t particleCount = 1;
  // Planar one-particle blocks: mass, x, y, z, vx, vy, vz.
  constexpr std::array<double, 7> particle{2.0, 1.0, 2.0, -1.0, 2.0, -3.0, 0.5};

  output.write(reinterpret_cast<const char *>(&particleCount),
               sizeof(particleCount));
  output.write(reinterpret_cast<const char *>(particle.data()),
               static_cast<std::streamsize>(sizeof(particle)));

  if (!output.good()) {
    std::cerr << "Could not write particle fixture: " << argv[1] << '\n';
    return 1;
  }

  return 0;
}
