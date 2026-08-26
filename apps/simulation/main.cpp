#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "particle.hpp"
#include "simulation.hpp"

namespace {

struct CommandLineOptions {
  std::string inputPath;
  std::size_t steps = 0;
  SimulationConfig simulation{0.0, 0.5, 1.0e-6};
  bool showHelp = false;
  bool hasTimeStep = false;
  bool hasSteps = false;
};

void printUsage(std::ostream &output, const char *program) {
  output << "Usage:\n"
         << "  " << program
         << " --input FILE --time-step DT --steps N [OPTIONS]\n\n"
         << "Required arguments:\n"
         << "  -i, --input FILE       Binary particle input file\n"
         << "  -t, --time-step DT     Positive shared simulation timestep\n"
         << "  -n, --steps N          Positive number of timesteps\n\n"
         << "Force options:\n"
         << "      --theta VALUE      Barnes-Hut opening angle (default: 0.5)\n"
         << "      --softening VALUE  Non-negative softening length "
            "(default: 1e-6)\n\n"
         << "Other options:\n"
         << "  -h, --help             Show this help and exit\n";
}

std::string_view requireValue(int argc, char **argv, int &index,
                              std::string_view option) {
  if (index + 1 >= argc)
    throw std::invalid_argument("Missing value for " + std::string(option));

  return argv[++index];
}

double parseDouble(std::string_view text, std::string_view option) {
  const std::string valueText(text);
  char *end = nullptr;
  errno = 0;
  const double value = std::strtod(valueText.c_str(), &end);

  if (end == valueText.c_str() || *end != '\0' || errno == ERANGE ||
      !std::isfinite(value)) {
    throw std::invalid_argument("Invalid numeric value for " +
                                std::string(option) + ": " + valueText);
  }

  return value;
}

std::size_t parseStepCount(std::string_view text, std::string_view option) {
  std::size_t value = 0;
  const char *first = text.data();
  const char *last = first + text.size();
  const auto result = std::from_chars(first, last, value);

  if (result.ec != std::errc{} || result.ptr != last || value == 0) {
    throw std::invalid_argument("Invalid positive integer for " +
                                std::string(option) + ": " + std::string(text));
  }

  return value;
}

CommandLineOptions parseCommandLine(int argc, char **argv) {
  CommandLineOptions options;

  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);

    if (option == "-h" || option == "--help") {
      options.showHelp = true;
      return options;
    }

    if (option == "-i" || option == "--input") {
      options.inputPath = requireValue(argc, argv, index, option);
    } else if (option == "-t" || option == "--time-step") {
      options.simulation.timeStep =
          parseDouble(requireValue(argc, argv, index, option), option);
      options.hasTimeStep = true;
    } else if (option == "-n" || option == "--steps") {
      options.steps =
          parseStepCount(requireValue(argc, argv, index, option), option);
      options.hasSteps = true;
    } else if (option == "--theta") {
      options.simulation.theta =
          parseDouble(requireValue(argc, argv, index, option), option);
    } else if (option == "--softening") {
      options.simulation.softening =
          parseDouble(requireValue(argc, argv, index, option), option);
    } else {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }

  if (options.inputPath.empty())
    throw std::invalid_argument("--input is required");
  if (!options.hasTimeStep)
    throw std::invalid_argument("--time-step is required");
  if (!options.hasSteps)
    throw std::invalid_argument("--steps is required");

  if (options.simulation.timeStep <= 0.0)
    throw std::invalid_argument("--time-step must be positive");
  if (options.simulation.theta < 0.0)
    throw std::invalid_argument("--theta must be non-negative");
  if (options.simulation.softening < 0.0)
    throw std::invalid_argument("--softening must be non-negative");

  return options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const CommandLineOptions options = parseCommandLine(argc, argv);

    if (options.showHelp) {
      printUsage(std::cout, argv[0]);
      return 0;
    }

    std::ifstream input(options.inputPath, std::ios::binary);
    if (!input.is_open())
      throw std::runtime_error("Could not open particle input file: " +
                               options.inputPath);

    Particles particles(input);
    Simulation simulation(std::move(particles), options.simulation);
    const std::size_t particleCount = simulation.particles().count;

    std::cout << "Loaded " << particleCount << " particles from "
              << options.inputPath << '\n';

    const auto start = std::chrono::steady_clock::now();
    simulation.initialize();
    simulation.run(options.steps);
    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finish - start;

    std::cout << std::setprecision(17) << "Simulation complete\n"
              << "  steps: " << simulation.stepNumber() << '\n'
              << "  simulated time: " << simulation.time() << '\n'
              << "  wall time: " << elapsed.count() << " seconds\n";
    return 0;
  } catch (const std::invalid_argument &error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Simulation failed: " << error.what() << '\n';
    return 1;
  }
}
