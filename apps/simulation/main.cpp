#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include "particle.hpp"
#include "simulation.hpp"
#include "streaming/device_capture.hpp"
#include "streaming/frame.hpp"
#include "streaming/network.hpp"
#include "streaming/server_process.hpp"

namespace {

struct CommandLineOptions {
  std::string inputPath;
  std::size_t steps = 0;
  SimulationConfig simulation{0.0, 0.5, 1.0e-6};
  nbody::streaming::ServerConfig server;
  nbody::streaming::RelaySourceConfig relay;
  std::filesystem::path serverExecutable;
  double sampleRate = 60.0;
  std::size_t maximumStreamParticles = 100000;
  std::chrono::milliseconds streamDrainTimeout{30000};
  bool cluster = false;
  bool relayEnabled = false;
  bool streamEnabled = false;
  bool showHelp = false;
  bool hasTimeStep = false;
  bool hasSteps = false;
};

void printUsage(std::ostream &output, const char *program) {
  output
      << "Usage:\n"
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
      << "Graphical stream options:\n"
      << "      --cluster          Write frames and start the stream server\n"
      << "      --stream-dir DIR   Durable frame queue; also enables output\n"
      << "      --sample-rate HZ   Maximum wall-clock sample rate (default: "
         "60)\n"
      << "      --stream-max-particles N  Display sample size; 0 sends all "
         "(default: 100000)\n"
      << "      --stream-bind ADDR Server bind address (default: 127.0.0.1)\n"
      << "      --stream-port PORT Server TCP port (default: 4747)\n"
      << "      --stream-relay-host HOST  Connect server outward to relay\n"
      << "      --stream-relay-port PORT  Relay source port (default: 4748)\n"
      << "      --stream-relay-token-file FILE  Private authentication token\n"
      << "      --stream-reconnect-ms MS  Relay retry delay (default: 1000)\n"
      << "      --stream-drain-ms MS  Final queue drain timeout (default: 30000)\n"
      << "      --stream-server FILE Override the server executable path\n\n"
      << "Other options:\n"
      << "      --profile-stages     Print wall time for simulation stages\n"
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

std::uint16_t parsePort(std::string_view text, std::string_view option) {
  unsigned int value = 0;
  const char *first = text.data();
  const char *last = first + text.size();
  const auto result = std::from_chars(first, last, value);

  if (result.ec != std::errc{} || result.ptr != last || value == 0 ||
      value > 65535) {
    throw std::invalid_argument("Invalid TCP port for " + std::string(option) +
                                ": " + std::string(text));
  }
  return static_cast<std::uint16_t>(value);
}

std::size_t parseNonNegativeCount(std::string_view text,
                                  std::string_view option) {
  std::size_t value = 0;
  const char *first = text.data();
  const char *last = first + text.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{} || result.ptr != last) {
    throw std::invalid_argument("Invalid non-negative integer for " +
                                std::string(option) + ": " + std::string(text));
  }
  return value;
}

std::chrono::milliseconds parseMilliseconds(std::string_view text,
                                            std::string_view option,
                                            unsigned long maximumValue = 60000) {
  unsigned long value = 0;
  const char *first = text.data();
  const char *last = first + text.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{} || result.ptr != last || value == 0 ||
      value > maximumValue) {
    throw std::invalid_argument("Invalid millisecond value for " +
                                std::string(option) + ": " + std::string(text));
  }
  return std::chrono::milliseconds(value);
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
    } else if (option == "--cluster") {
      options.cluster = true;
      options.streamEnabled = true;
    } else if (option == "--stream-dir") {
      options.server.spoolDirectory = requireValue(argc, argv, index, option);
      options.streamEnabled = true;
    } else if (option == "--sample-rate") {
      options.sampleRate =
          parseDouble(requireValue(argc, argv, index, option), option);
    } else if (option == "--stream-max-particles") {
      options.maximumStreamParticles = parseNonNegativeCount(
          requireValue(argc, argv, index, option), option);
    } else if (option == "--stream-bind") {
      options.server.bindAddress = requireValue(argc, argv, index, option);
    } else if (option == "--stream-port") {
      options.server.port =
          parsePort(requireValue(argc, argv, index, option), option);
    } else if (option == "--stream-relay-host") {
      options.relay.host = requireValue(argc, argv, index, option);
      options.relayEnabled = true;
    } else if (option == "--stream-relay-port") {
      options.relay.port =
          parsePort(requireValue(argc, argv, index, option), option);
      options.relayEnabled = true;
    } else if (option == "--stream-relay-token-file") {
      options.relay.tokenFile = requireValue(argc, argv, index, option);
      options.relayEnabled = true;
    } else if (option == "--stream-reconnect-ms") {
      options.relay.reconnectInterval =
          parseMilliseconds(requireValue(argc, argv, index, option), option);
      options.relayEnabled = true;
    } else if (option == "--stream-drain-ms") {
      constexpr unsigned long maximumDrainTimeoutMs = 24UL * 60UL * 60UL * 1000UL;
      options.streamDrainTimeout = parseMilliseconds(
          requireValue(argc, argv, index, option), option,
          maximumDrainTimeoutMs);
    } else if (option == "--stream-server") {
      options.serverExecutable = requireValue(argc, argv, index, option);
    } else if (option == "--profile-stages") {
      options.simulation.profileStages = true;
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
  if (options.sampleRate <= 0.0)
    throw std::invalid_argument("--sample-rate must be positive");
  if (options.server.spoolDirectory.empty())
    throw std::invalid_argument("--stream-dir must not be empty");
  if (options.server.bindAddress.empty())
    throw std::invalid_argument("--stream-bind must not be empty");
  if (options.relayEnabled && !options.cluster)
    throw std::invalid_argument(
        "Stream relay options require --cluster automatic server startup");
  if (options.relayEnabled && options.relay.host.empty())
    throw std::invalid_argument(
        "--stream-relay-host is required when relay options are used");
  if (options.relayEnabled && options.relay.tokenFile.empty())
    throw std::invalid_argument(
        "--stream-relay-token-file is required in relay mode");

  return options;
}

std::size_t queuedFrameCount(const std::filesystem::path &directory) {
  std::error_code error;
  std::filesystem::directory_iterator entries(directory, error);
  if (error)
    throw std::runtime_error("Could not inspect frame queue '" +
                             directory.string() + "': " + error.message());
  std::size_t count = 0;
  for (const auto &entry : entries) {
    if (entry.is_regular_file(error) && !error &&
        entry.path().extension() == ".nbsnap")
      ++count;
    error.clear();
  }
  return count;
}

void drainFrameQueue(const std::filesystem::path &directory,
                     std::chrono::milliseconds timeout) {
  std::size_t remaining = queuedFrameCount(directory);
  if (remaining == 0)
    return;

  std::cout << "Waiting up to " << timeout.count()
            << " ms for the stream client to acknowledge " << remaining
            << " queued frames\n"
            << std::flush;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    remaining = queuedFrameCount(directory);
  } while (remaining > 0 && std::chrono::steady_clock::now() < deadline);

  if (remaining == 0) {
    std::cout << "Frame queue drained\n" << std::flush;
  } else {
    std::cerr << "Stream drain timed out with " << remaining
              << " durable frames still queued in " << directory << '\n'
              << std::flush;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const CommandLineOptions options = parseCommandLine(argc, argv);

    if (options.showHelp) {
      printUsage(std::cout, argv[0]);
      return 0;
    }

    std::unique_ptr<nbody::streaming::FrameSpool> frameSpool;
    std::unique_ptr<nbody::streaming::ServerProcess> streamServer;
    std::unique_ptr<nbody::streaming::FrameSampler> frameSampler;
    std::size_t framesWritten = 0;

    if (options.streamEnabled) {
      frameSpool = std::make_unique<nbody::streaming::FrameSpool>(
          options.server.spoolDirectory);
      frameSampler =
          std::make_unique<nbody::streaming::FrameSampler>(options.sampleRate);
    }
    if (options.cluster) {
      const std::filesystem::path serverExecutable =
          options.serverExecutable.empty()
              ? nbody::streaming::defaultServerExecutable(argv[0])
              : options.serverExecutable;
      const std::optional<nbody::streaming::RelaySourceConfig> relayConfig =
          options.relayEnabled
              ? std::optional<nbody::streaming::RelaySourceConfig>(
                    options.relay)
              : std::nullopt;
      streamServer = std::make_unique<nbody::streaming::ServerProcess>(
          serverExecutable, options.server, relayConfig, std::cout);
    }

    // Start the separate server before the first CUDA allocation. Forking a
    // process after CUDA has initialized is unsupported by the CUDA runtime.
    std::ifstream input(options.inputPath, std::ios::binary);
    if (!input.is_open())
      throw std::runtime_error("Could not open particle input file: " +
                               options.inputPath);

    Particles particles(input);
    Simulation simulation(std::move(particles), options.simulation);
    const DeviceParticlesView mutableView = simulation.particles();
    const ConstDeviceParticlesView particlesView{
        mutableView.count, mutableView.mass, mutableView.x,    mutableView.y,
        mutableView.z,     mutableView.vx,   mutableView.vy,   mutableView.vz,
        mutableView.type};
    const std::size_t particleCount = particlesView.count;
    std::unique_ptr<nbody::streaming::DeviceFrameWriter> deviceFrameWriter;
    if (frameSpool) {
      deviceFrameWriter = std::make_unique<nbody::streaming::DeviceFrameWriter>(
          particlesView, options.maximumStreamParticles);
    }

    std::cout << "Loaded " << particleCount << " particles from "
              << options.inputPath << '\n'
              << std::flush;

    const auto start = std::chrono::steady_clock::now();
    const auto sampleClockStart = start;
    std::size_t lastCapturedStep = std::numeric_limits<std::size_t>::max();

    auto captureFrame = [&](bool force) {
      if (!frameSampler)
        return;
      const std::chrono::duration<double> sampleElapsed =
          std::chrono::steady_clock::now() - sampleClockStart;
      if (!force && !frameSampler->isDue(sampleElapsed.count()))
        return;

      deviceFrameWriter->write(*frameSpool, frameSampler->takeSequence(),
                               simulation.stepNumber(), simulation.time(),
                               options.steps);
      ++framesWritten;
      lastCapturedStep = simulation.stepNumber();
    };

    // Publish the input positions before the first force-tree construction.
    // Initialization can take minutes for million-particle datasets, while
    // frame zero only needs a bounded device-to-host sample. This guarantees
    // that a connected viewer receives useful data even if initialization
    // consumes the remainder of a short batch allocation.
    captureFrame(false);
    simulation.initialize();
    for (std::size_t step = 0; step < options.steps; ++step) {
      simulation.step();
      captureFrame(false);
    }
    if (frameSampler && lastCapturedStep != simulation.stepNumber())
      captureFrame(true);
    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finish - start;

    std::cout << std::setprecision(17) << "Simulation complete\n"
              << "  steps: " << simulation.stepNumber() << '\n'
              << "  simulated time: " << simulation.time() << '\n'
              << "  wall time: " << elapsed.count() << " seconds\n";
    if (frameSpool) {
      std::cout << "  frames written: " << framesWritten << '\n'
                << "  particles per frame: "
                << deviceFrameWriter->sampleParticleCount() << " of "
                << particleCount << '\n'
                << "  frame spool: " << frameSpool->directory() << '\n';
    }
    if (streamServer && frameSpool)
      drainFrameQueue(frameSpool->directory(), options.streamDrainTimeout);
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
