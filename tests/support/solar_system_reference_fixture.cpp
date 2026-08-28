#include "solar_system_reference_fixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

constexpr double kGravitationalConstant = 6.67430e-11;
constexpr std::size_t kStateComponentCount = 6;
using State = std::array<double, kStateComponentCount>;

struct BodyReference {
  double gmKilometersCubedPerSecondSquared;
  State epochA;
  State epochB;
};

// NASA/JPL HORIZONS DE441 geometric barycentric ICRF vectors. Positions and
// velocities are in SI units at JD TDB 2461278.5 and 2461279.5 respectively.
// The ordering matches download_solar_system.py: Sun, then increasing planet
// SPK-ID (199 through 999).
constexpr std::array<BodyReference, 10> kBodies{{
    {132712440041.27942,
     {-219293369.9744537, -723192502.497016, -297351032.2718988,
      10.724736401600019, 3.672014100288164, 1.3479177130510591},
     {-218367066.7455701, -722874780.0581535, -297234367.37209404,
      10.71747428014971, 3.682668321493552, 1.352655920272148}},
    {22031.868551400003,
     {-42416455671.5933, 23284288865.09411, 16900766491.925678,
      -37453.96158560444, -35154.429930698345, -14897.5322584611},
     {-45498139309.223564, 20164895022.20486, 15553792248.91103,
      -33867.497320521914, -36996.60169537039, -16253.32364584914}},
    {324858.592,
     {48293564533.46542, -88465843354.88869, -42848283899.569214,
      31123.96547434121, 14845.8185316058, 4711.517770624347},
     {50963734940.90144, -87149693269.70238, -42424949322.405624,
      30681.59712394868, 15618.65469686081, 5087.259139079727}},
    {398600.43550702266,
     {133801656387.7991, -64928704778.73579, -28129952692.32178,
      13306.49074081925, 24122.069555855313, 10455.01063171265},
     {134932145654.9542, -62835531432.78946, -27222730469.07037,
      12861.644463554929, 24329.9349819525, 10545.043922288769}},
    {42828.37485735626,
     {88819379438.3502, 189190739254.0228, 84410328953.501, -21343.258484898863,
      10340.318359720679, 5318.4643175086185},
     {86971540552.89006, 190076010492.06097, 84866216146.5035,
      -21430.44733164364, 10152.00682664727, 5234.441117913465}},
    {126686531.9003704,
     {-490614619920.945, 566644308210.9731, 254828972666.59518,
      -10410.4944215994, -6970.216337904196, -2734.206722038477},
     {-491513520804.0385, 566041552850.423, 254592501707.56598,
      -10398.407680232489, -6981.924257983896, -2739.419758016719}},
    {37931206.23436167,
     {1391911951588.3738, 238371066973.2882, 38506352122.24183,
      -2133.293581799375, 8752.012368916608, 3707.1612455509307},
     {1391727401505.878, 239127181531.456, 38826644230.853516,
      -2138.695859562282, 8750.655112584991, 3707.005907179126}},
    {5793950.610340896,
     {1354767405675.178, 2364554028380.607, 1016448210578.656,
      -6076.221625702208, 2583.506689751777, 1217.4086392151212},
     {1354242398051.1528, 2364777196613.7197, 1016553366688.476,
      -6076.804306892515, 2582.453708346579, 1216.712882379983}},
    {6835099.968446816,
     {4464307857491.136, 215608251771.4135, -22895981431.21723,
      -265.8175265267751, 5054.189828591928, 2076.070340500793},
     {4464284898402.725, 216044957052.0731, -22716603239.85519,
      -265.612463426985, 5054.681320998454, 2076.047980772381}},
    {869.3261226311508,
     {2971940929406.876, -3882404000396.431, -2107014519979.2148,
      4676.586865698008, 2307.0566256260418, -688.4236557383999},
     {2972344826474.254, -3882204940445.643, -2107074944182.348,
      4670.266124028556, 2298.630180356165, -708.5605640010033}},
}};

void writeBlock(std::ofstream &output,
                const std::array<double, kBodies.size()> &values) {
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(double)));
}

void writeDataset(const std::filesystem::path &path, bool useEpochB) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open())
    throw std::runtime_error("Could not create solar-system fixture: " +
                             path.string());

  const auto particleCount = static_cast<std::uint64_t>(kBodies.size());
  output.write(reinterpret_cast<const char *>(&particleCount),
               sizeof(particleCount));

  std::array<double, kBodies.size()> block{};
  for (std::size_t index = 0; index < kBodies.size(); ++index) {
    block[index] = kBodies[index].gmKilometersCubedPerSecondSquared * 1.0e9 /
                   kGravitationalConstant;
  }
  writeBlock(output, block);

  for (std::size_t component = 0; component < kStateComponentCount;
       ++component) {
    for (std::size_t index = 0; index < kBodies.size(); ++index) {
      const State &state =
          useEpochB ? kBodies[index].epochB : kBodies[index].epochA;
      block[index] = state[component];
    }
    writeBlock(output, block);
  }

  if (!output.good())
    throw std::runtime_error("Could not write solar-system fixture: " +
                             path.string());
}

} // namespace

SolarSystemReferenceDatasets writeSolarSystemReferenceDatasets(
    const std::filesystem::path &outputDirectory) {
  std::filesystem::create_directories(outputDirectory);
  const SolarSystemReferenceDatasets datasets{
      outputDirectory / "ephemeris_reference_epoch_a.bin",
      outputDirectory / "ephemeris_reference_epoch_b.bin"};
  writeDataset(datasets.epochA, false);
  writeDataset(datasets.epochB, true);
  return datasets;
}
