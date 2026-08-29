#include "streaming/frame.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace nbody::streaming {
namespace {

constexpr std::array<char, 8> kFrameMagic{'N', 'B', 'S', 'N',
                                           'A', 'P', '0', '1'};
template <typename Unsigned>
void writeLittleEndian(std::ostream &output, Unsigned value) {
  static_assert(std::is_unsigned<Unsigned>::value,
                "only unsigned integers are supported");
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    const auto current = static_cast<char>((value >> (byte * 8U)) & 0xffU);
    output.put(current);
  }
}

template <typename Unsigned> Unsigned readLittleEndian(std::istream &input) {
  static_assert(std::is_unsigned<Unsigned>::value,
                "only unsigned integers are supported");
  Unsigned value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
    const int current = input.get();
    if (current == std::char_traits<char>::eof())
      throw std::runtime_error("Truncated N-body frame header");
    value |= static_cast<Unsigned>(static_cast<unsigned char>(current))
             << (byte * 8U);
  }
  return value;
}

void writeDouble(std::ostream &output, double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "the frame format requires 64-bit doubles");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  writeLittleEndian(output, bits);
}

double readDouble(std::istream &input) {
  const std::uint64_t bits = readLittleEndian<std::uint64_t>(input);
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void writeHeader(std::ostream &output, const FrameHeader &header) {
  output.write(kFrameMagic.data(),
               static_cast<std::streamsize>(kFrameMagic.size()));
  writeLittleEndian(output, header.version);
  writeLittleEndian(output, static_cast<std::uint32_t>(kFrameHeaderBytes));
  writeLittleEndian(output, header.sequence);
  writeLittleEndian(output, header.simulationStep);
  writeDouble(output, header.simulationTime);
  writeLittleEndian(output, header.sourceParticleCount);
  writeLittleEndian(output, header.particleCount);
  writeLittleEndian(output, header.scalarBytes);
  writeLittleEndian(output, header.components);
  writeLittleEndian(output, header.payloadBytes);
}

std::string makeSessionId() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(now).count();
#if defined(__unix__) || defined(__APPLE__)
  const auto process = static_cast<unsigned long long>(::getpid());
#else
  const auto process = 0ULL;
#endif
  return std::to_string(microseconds) + "-" + std::to_string(process);
}

std::string makeFrameName(const std::string &sessionId,
                          std::uint64_t sequence) {
  std::ostringstream name;
  name << "run-" << sessionId << "-frame-" << std::setw(20)
       << std::setfill('0') << sequence << ".nbsnap";
  return name.str();
}

void validateSessionId(const std::string &sessionId) {
  if (sessionId.empty())
    throw std::invalid_argument("Frame session id cannot be empty");
  if (!std::all_of(sessionId.begin(), sessionId.end(), [](char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9') || character == '-' ||
               character == '_';
      })) {
    throw std::invalid_argument(
        "Frame session id may contain only letters, numbers, '-' and '_'");
  }
}

} // namespace

FrameSampler::FrameSampler(double samplesPerSecond)
    : rate_(samplesPerSecond), interval_(1.0 / samplesPerSecond) {
  if (!std::isfinite(samplesPerSecond) || samplesPerSecond <= 0.0) {
    throw std::invalid_argument("Frame sample rate must be finite and positive");
  }
}

bool FrameSampler::isDue(double elapsedWallSeconds) {
  if (!std::isfinite(elapsedWallSeconds) || elapsedWallSeconds < 0.0)
    throw std::invalid_argument(
        "Frame sampling time must be finite and non-negative");

  const double tolerance =
      std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(elapsedWallSeconds)) * 8.0;
  if (elapsedWallSeconds + tolerance < nextSampleTime_)
    return false;

  const double elapsed =
      std::max(0.0, elapsedWallSeconds - nextSampleTime_);
  const double skippedIntervals = std::floor(elapsed / interval_);
  nextSampleTime_ += (skippedIntervals + 1.0) * interval_;
  return true;
}

std::uint64_t FrameSampler::takeSequence() noexcept { return sequence_++; }

double FrameSampler::rate() const noexcept { return rate_; }

FrameSpool::FrameSpool(std::filesystem::path directory, std::string sessionId)
    : directory_(std::move(directory)),
      sessionId_(sessionId.empty() ? makeSessionId() : std::move(sessionId)) {
  if (directory_.empty())
    throw std::invalid_argument("Frame spool directory cannot be empty");
  validateSessionId(sessionId_);

  std::error_code error;
  std::filesystem::create_directories(directory_, error);
  if (error)
    throw std::runtime_error("Could not create frame spool directory '" +
                             directory_.string() + "': " + error.message());
}

const std::filesystem::path &FrameSpool::directory() const noexcept {
  return directory_;
}

const std::string &FrameSpool::sessionId() const noexcept { return sessionId_; }

std::filesystem::path
FrameSpool::writePositions(std::uint64_t sequence,
                           std::uint64_t simulationStep,
                           double simulationTime, std::size_t particleCount,
                           const PositionChunkReader &reader,
                           std::size_t sourceParticleCount) const {
  if (!std::isfinite(simulationTime) || simulationTime < 0.0)
    throw std::invalid_argument("Frame simulation time is invalid");
  if (particleCount == 0)
    throw std::invalid_argument("Cannot write an empty position frame");
  if (!reader)
    throw std::invalid_argument("Position frame reader is not callable");
  if (sourceParticleCount == 0)
    sourceParticleCount = particleCount;
  if (sourceParticleCount < particleCount)
    throw std::invalid_argument(
        "Frame source particle count cannot be smaller than its sample");
  if (particleCount >
      std::numeric_limits<std::uint64_t>::max() /
          (kPositionComponents * sizeof(float))) {
    throw std::overflow_error("Position frame payload is too large");
  }

  const auto payloadBytes =
      static_cast<std::uint64_t>(particleCount) * kPositionComponents *
      sizeof(float);
  const FrameHeader header{1,
                           sequence,
                           simulationStep,
                           simulationTime,
                           static_cast<std::uint64_t>(sourceParticleCount),
                           static_cast<std::uint64_t>(particleCount),
                           sizeof(float),
                           kPositionComponents,
                           payloadBytes};

  const std::filesystem::path finalPath =
      directory_ / makeFrameName(sessionId_, sequence);
  std::filesystem::path temporaryPath = finalPath;
  temporaryPath += ".part";

  std::error_code error;
  if (std::filesystem::exists(finalPath, error) && !error)
    throw std::runtime_error("Refusing to overwrite queued frame: " +
                             finalPath.string());

  try {
    std::ofstream output(temporaryPath,
                         std::ios::binary | std::ios::trunc | std::ios::out);
    if (!output.is_open())
      throw std::runtime_error("Could not create temporary frame: " +
                               temporaryPath.string());

    writeHeader(output, header);
    std::vector<float> positions(
        std::min(kFrameChunkParticles, particleCount) * kPositionComponents);

    for (std::size_t offset = 0; offset < particleCount;) {
      const std::size_t count =
          std::min(kFrameChunkParticles, particleCount - offset);
      reader(offset, count, positions.data());
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      for (std::size_t valueIndex = 0;
           valueIndex < count * kPositionComponents; ++valueIndex) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &positions[valueIndex], sizeof(bits));
        bits = ((bits & 0x000000ffU) << 24U) |
               ((bits & 0x0000ff00U) << 8U) |
               ((bits & 0x00ff0000U) >> 8U) |
               ((bits & 0xff000000U) >> 24U);
        std::memcpy(&positions[valueIndex], &bits, sizeof(bits));
      }
#endif
      const auto bytes = static_cast<std::streamsize>(
          count * kPositionComponents * sizeof(float));
      output.write(reinterpret_cast<const char *>(positions.data()), bytes);
      if (!output)
        throw std::runtime_error("Failed while writing frame payload: " +
                                 temporaryPath.string());
      offset += count;
    }

    output.flush();
    if (!output)
      throw std::runtime_error("Failed to flush frame: " +
                               temporaryPath.string());
    output.close();

    std::filesystem::rename(temporaryPath, finalPath, error);
    if (error)
      throw std::runtime_error("Could not publish completed frame '" +
                               finalPath.string() + "': " + error.message());
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporaryPath, ignored);
    throw;
  }

  return finalPath;
}

FrameHeader readFrameHeader(const std::filesystem::path &framePath) {
  std::ifstream input(framePath, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open N-body frame: " +
                             framePath.string());

  std::array<char, kFrameMagic.size()> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kFrameMagic)
    throw std::runtime_error("Invalid N-body frame magic: " +
                             framePath.string());

  FrameHeader header;
  header.version = readLittleEndian<std::uint32_t>(input);
  const auto headerBytes = readLittleEndian<std::uint32_t>(input);
  header.sequence = readLittleEndian<std::uint64_t>(input);
  header.simulationStep = readLittleEndian<std::uint64_t>(input);
  header.simulationTime = readDouble(input);
  header.sourceParticleCount = readLittleEndian<std::uint64_t>(input);
  header.particleCount = readLittleEndian<std::uint64_t>(input);
  header.scalarBytes = readLittleEndian<std::uint32_t>(input);
  header.components = readLittleEndian<std::uint32_t>(input);
  header.payloadBytes = readLittleEndian<std::uint64_t>(input);

  const bool payloadWouldOverflow =
      header.particleCount >
      std::numeric_limits<std::uint64_t>::max() /
          (kPositionComponents * sizeof(float));
  if (header.version != 1 || headerBytes != kFrameHeaderBytes ||
      header.scalarBytes != sizeof(float) ||
      header.components != kPositionComponents || header.particleCount == 0 ||
      header.sourceParticleCount < header.particleCount ||
      !std::isfinite(header.simulationTime) || header.simulationTime < 0.0 ||
      payloadWouldOverflow ||
      header.payloadBytes != header.particleCount * header.components *
                                 header.scalarBytes) {
    throw std::runtime_error("Unsupported or inconsistent N-body frame header: " +
                             framePath.string());
  }

  std::error_code error;
  const auto actualBytes = std::filesystem::file_size(framePath, error);
  if (error || actualBytes != kFrameHeaderBytes + header.payloadBytes)
    throw std::runtime_error("N-body frame size does not match its header: " +
                             framePath.string());
  return header;
}

} // namespace nbody::streaming
