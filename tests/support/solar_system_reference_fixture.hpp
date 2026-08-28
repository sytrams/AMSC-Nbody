#pragma once

#include <filesystem>

struct SolarSystemReferenceDatasets {
  std::filesystem::path epochA;
  std::filesystem::path epochB;
};

// Write the compact Sun-and-planets HORIZONS fixture in the same planar binary
// format produced by data/download_solar_system.py.
SolarSystemReferenceDatasets
writeSolarSystemReferenceDatasets(const std::filesystem::path &outputDirectory);
