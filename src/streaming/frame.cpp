#include "streaming/frame.hpp"

#include "particle_type.hpp"

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
  writeLittleEndian(output, header.headerBytes);
  writeLittleEndian(output, header.sequence);
  writeLittleEndian(output, header.simulationStep);
  writeDouble(output, header.simulationTime);
  writeLittleEndian(output, header.sourceParticleCount);
  writeLittleEndian(output, header.particleCount);
  writeLittleEndian(output, header.scalarBytes);
  writeLittleEndian(output, header.components);
  writeLittleEndian(output, header.payloadBytes);
  if (header.version == kLegacyTypedFrameVersion &&
      header.headerBytes == kLegacyTypedFrameHeaderBytes) {
    writeLittleEndian(output, header.totalSteps);
  } else if ((header.version == kQuantizedFrameVersion &&
              header.headerBytes == kQuantizedFrameHeaderBytes) ||
             (header.version == kFrameVersion &&
              header.headerBytes == kFrameHeaderBytes)) {
    for (const double value : header.bounds.minimum)
      writeDouble(output, value);
    for (const double value : header.bounds.maximum)
      writeDouble(output, value);
    if (header.version == kFrameVersion)
      writeLittleEndian(output, header.particleTypeBytes);
  }
}

void validateBounds(const PositionBounds &bounds) {
  for (std::size_t component = 0; component < kPositionComponents;
       ++component) {
    if (!std::isfinite(bounds.minimum[component]) ||
        !std::isfinite(bounds.maximum[component]) ||
        bounds.minimum[component] > bounds.maximum[component] ||
        !std::isfinite(bounds.maximum[component] -
                       bounds.minimum[component])) {
      throw std::invalid_argument(
          "Position frame bounds must be finite and ordered");
    }
  }
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

std::string makeCompletionName(const std::string &sessionId) {
  return "run-" + sessionId + ".complete";
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
  return writeTypedPositions(sequence, simulationStep, simulationTime, 0,
                             particleCount, reader, {}, sourceParticleCount);
}

std::filesystem::path FrameSpool::writeTypedPositions(
    std::uint64_t sequence, std::uint64_t simulationStep,
    double simulationTime, std::uint64_t totalSteps,
    std::size_t particleCount, const PositionChunkReader &positionReader,
    const ParticleTypeChunkReader &typeReader,
    std::size_t sourceParticleCount) const {
  if (!std::isfinite(simulationTime) || simulationTime < 0.0)
    throw std::invalid_argument("Frame simulation time is invalid");
  if (typeReader && totalSteps < simulationStep)
    throw std::invalid_argument(
        "Frame total step count cannot be smaller than its current step");
  if (particleCount == 0)
    throw std::invalid_argument("Cannot write an empty position frame");
  if (!positionReader)
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

  const auto positionBytes =
      static_cast<std::uint64_t>(particleCount) * kPositionComponents *
      sizeof(float);
  const auto particleTypeBytes =
      typeReader ? static_cast<std::uint64_t>(particleCount) : 0ULL;
  FrameHeader header;
  header.version = typeReader ? kLegacyTypedFrameVersion : kLegacyFrameVersion;
  header.headerBytes = static_cast<std::uint32_t>(
      typeReader ? kLegacyTypedFrameHeaderBytes : kLegacyFrameHeaderBytes);
  header.sequence = sequence;
  header.simulationStep = simulationStep;
  header.simulationTime = simulationTime;
  header.sourceParticleCount = static_cast<std::uint64_t>(sourceParticleCount);
  header.particleCount = static_cast<std::uint64_t>(particleCount);
  header.scalarBytes = sizeof(float);
  header.payloadBytes = positionBytes + particleTypeBytes;
  header.totalSteps = typeReader ? totalSteps : 0;
  header.particleTypeBytes = particleTypeBytes;

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
      positionReader(offset, count, positions.data());
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
      output.write(
          reinterpret_cast<const char *>(positions.data()),
          static_cast<std::streamsize>(count * kPositionComponents *
                                       sizeof(float)));
      if (!output)
        throw std::runtime_error("Failed while writing frame payload: " +
                                 temporaryPath.string());
      offset += count;
    }

    if (typeReader) {
      std::vector<std::uint8_t> types(
          std::min(kFrameChunkParticles, particleCount));
      for (std::size_t offset = 0; offset < particleCount;) {
        const std::size_t count =
            std::min(kFrameChunkParticles, particleCount - offset);
        typeReader(offset, count, types.data());
        for (std::size_t index = 0; index < count; ++index) {
          if (!isValidParticleType(types[index]))
            throw std::runtime_error(
                "Particle type reader returned an unsupported value");
        }
        output.write(reinterpret_cast<const char *>(types.data()),
                     static_cast<std::streamsize>(count));
        if (!output)
          throw std::runtime_error(
              "Failed while writing frame particle types: " +
              temporaryPath.string());
        offset += count;
      }
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

std::filesystem::path
FrameSpool::writeQuantizedPositions(
    std::uint64_t sequence, std::uint64_t simulationStep,
    double simulationTime, std::size_t particleCount,
    const PositionBounds &bounds, const QuantizedPositionChunkReader &reader,
    std::size_t sourceParticleCount,
    const ParticleTypeChunkReader &typeReader) const {
  if (!std::isfinite(simulationTime) || simulationTime < 0.0)
    throw std::invalid_argument("Frame simulation time is invalid");
  if (particleCount == 0)
    throw std::invalid_argument("Cannot write an empty position frame");
  if (!reader)
    throw std::invalid_argument("Position frame reader is not callable");
  validateBounds(bounds);
  if (sourceParticleCount == 0)
    sourceParticleCount = particleCount;
  if (sourceParticleCount < particleCount)
    throw std::invalid_argument(
        "Frame source particle count cannot be smaller than its sample");
  if (particleCount >
      std::numeric_limits<std::uint64_t>::max() /
          (kPositionComponents * sizeof(std::uint16_t))) {
    throw std::overflow_error("Position frame payload is too large");
  }

  const auto positionBytes =
      static_cast<std::uint64_t>(particleCount) * kPositionComponents *
      sizeof(std::uint16_t);
  const auto particleTypeBytes =
      typeReader ? static_cast<std::uint64_t>(particleCount) : 0ULL;
  if (positionBytes > std::numeric_limits<std::uint64_t>::max() -
                          particleTypeBytes) {
    throw std::overflow_error("Position frame payload is too large");
  }
  FrameHeader header;
  if (!typeReader) {
    header.version = kQuantizedFrameVersion;
    header.headerBytes =
        static_cast<std::uint32_t>(kQuantizedFrameHeaderBytes);
  }
  header.sequence = sequence;
  header.simulationStep = simulationStep;
  header.simulationTime = simulationTime;
  header.sourceParticleCount = static_cast<std::uint64_t>(sourceParticleCount);
  header.particleCount = static_cast<std::uint64_t>(particleCount);
  header.payloadBytes = positionBytes + particleTypeBytes;
  header.bounds = bounds;
  header.particleTypeBytes = particleTypeBytes;

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
    std::vector<std::uint16_t> positions(
        std::min(kFrameChunkParticles, particleCount) * kPositionComponents);

    for (std::size_t offset = 0; offset < particleCount;) {
      const std::size_t count =
          std::min(kFrameChunkParticles, particleCount - offset);
      reader(offset, count, positions.data());
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      for (std::size_t valueIndex = 0;
           valueIndex < count * kPositionComponents; ++valueIndex) {
        const std::uint16_t value = positions[valueIndex];
        positions[valueIndex] = static_cast<std::uint16_t>(
            (value << 8U) | (value >> 8U));
      }
#endif
      const auto bytes = static_cast<std::streamsize>(
          count * kPositionComponents * sizeof(std::uint16_t));
      output.write(reinterpret_cast<const char *>(positions.data()), bytes);
      if (!output)
        throw std::runtime_error("Failed while writing frame payload: " +
                                 temporaryPath.string());
      offset += count;
    }

    if (typeReader) {
      std::vector<std::uint8_t> types(
          std::min(kFrameChunkParticles, particleCount));
      for (std::size_t offset = 0; offset < particleCount;) {
        const std::size_t count =
            std::min(kFrameChunkParticles, particleCount - offset);
        typeReader(offset, count, types.data());
        for (std::size_t index = 0; index < count; ++index) {
          if (!isValidParticleType(types[index])) {
            throw std::runtime_error(
                "Particle type reader returned an unsupported value");
          }
        }
        output.write(reinterpret_cast<const char *>(types.data()),
                     static_cast<std::streamsize>(count));
        if (!output)
          throw std::runtime_error(
              "Failed while writing frame particle types: " +
              temporaryPath.string());
        offset += count;
      }
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

std::filesystem::path FrameSpool::markComplete() const {
  const std::filesystem::path finalPath =
      directory_ / makeCompletionName(sessionId_);
  std::filesystem::path temporaryPath = finalPath;
  temporaryPath += ".part";

  std::error_code error;
  if (std::filesystem::exists(finalPath, error) && !error)
    return finalPath;

  try {
    std::ofstream output(temporaryPath,
                         std::ios::out | std::ios::trunc | std::ios::binary);
    if (!output.is_open())
      throw std::runtime_error("Could not create completion marker: " +
                               temporaryPath.string());
    output << sessionId_ << '\n';
    output.flush();
    if (!output)
      throw std::runtime_error("Could not flush completion marker: " +
                               temporaryPath.string());
    output.close();
    std::filesystem::rename(temporaryPath, finalPath, error);
    if (error)
      throw std::runtime_error("Could not publish completion marker '" +
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
  header.headerBytes = readLittleEndian<std::uint32_t>(input);
  header.sequence = readLittleEndian<std::uint64_t>(input);
  header.simulationStep = readLittleEndian<std::uint64_t>(input);
  header.simulationTime = readDouble(input);
  header.sourceParticleCount = readLittleEndian<std::uint64_t>(input);
  header.particleCount = readLittleEndian<std::uint64_t>(input);
  header.scalarBytes = readLittleEndian<std::uint32_t>(input);
  header.components = readLittleEndian<std::uint32_t>(input);
  header.payloadBytes = readLittleEndian<std::uint64_t>(input);

  const bool legacyLayout =
      header.version == kLegacyFrameVersion &&
      header.headerBytes == kLegacyFrameHeaderBytes;
  const bool legacyTypedLayout =
      header.version == kLegacyTypedFrameVersion &&
      header.headerBytes == kLegacyTypedFrameHeaderBytes;
  const bool quantizedLayout =
      header.version == kQuantizedFrameVersion &&
      header.headerBytes == kQuantizedFrameHeaderBytes;
  const bool typedLayout = header.version == kFrameVersion &&
                           header.headerBytes == kFrameHeaderBytes;

  if (legacyTypedLayout) {
    header.totalSteps = readLittleEndian<std::uint64_t>(input);
    header.particleTypeBytes = header.particleCount;
  } else if (quantizedLayout || typedLayout) {
    for (double &value : header.bounds.minimum)
      value = readDouble(input);
    for (double &value : header.bounds.maximum)
      value = readDouble(input);
    if (typedLayout)
      header.particleTypeBytes = readLittleEndian<std::uint64_t>(input);
  }

  const bool legacy = legacyLayout && header.scalarBytes == sizeof(float);
  const bool legacyTyped =
      legacyTypedLayout && header.scalarBytes == sizeof(float) &&
      header.totalSteps >= header.simulationStep;
  const bool quantized = quantizedLayout &&
                         header.scalarBytes == sizeof(std::uint16_t);
  const bool typed = typedLayout &&
                     header.scalarBytes == sizeof(std::uint16_t) &&
                     header.particleTypeBytes == header.particleCount;
  const bool payloadWouldOverflow =
      header.scalarBytes == 0 ||
      header.particleCount >
          std::numeric_limits<std::uint64_t>::max() /
              (kPositionComponents *
               static_cast<std::uint64_t>(header.scalarBytes));
  const std::uint64_t expectedPositionBytes =
      (!legacy && !legacyTyped && !quantized && !typed) ||
              payloadWouldOverflow
          ? 0
          : header.particleCount * kPositionComponents * header.scalarBytes;
  const bool typedPayloadWouldOverflow =
      (legacyTyped || typed) &&
      expectedPositionBytes > std::numeric_limits<std::uint64_t>::max() -
                                  header.particleTypeBytes;
  const std::uint64_t expectedPayloadBytes =
      typedPayloadWouldOverflow
          ? 0
          : expectedPositionBytes + header.particleTypeBytes;
  const bool frameSizeWouldOverflow =
      header.payloadBytes > std::numeric_limits<std::uint64_t>::max() -
                                header.headerBytes;
  if ((!legacy && !legacyTyped && !quantized && !typed) ||
      header.components != kPositionComponents || header.particleCount == 0 ||
      header.sourceParticleCount < header.particleCount ||
      !std::isfinite(header.simulationTime) || header.simulationTime < 0.0 ||
      payloadWouldOverflow || typedPayloadWouldOverflow ||
      frameSizeWouldOverflow ||
      header.payloadBytes != expectedPayloadBytes) {
    throw std::runtime_error("Unsupported or inconsistent N-body frame header: " +
                             framePath.string());
  }
  if (quantized || typed) {
    try {
      validateBounds(header.bounds);
    } catch (const std::invalid_argument &) {
      throw std::runtime_error(
          "Unsupported or inconsistent N-body frame header: " +
          framePath.string());
    }
  }

  std::error_code error;
  const auto actualBytes = std::filesystem::file_size(framePath, error);
  if (error || actualBytes != header.headerBytes + header.payloadBytes)
    throw std::runtime_error("N-body frame size does not match its header: " +
                             framePath.string());
  return header;
}

std::vector<float>
readFramePositions(const std::filesystem::path &framePath) {
  const FrameHeader header = readFrameHeader(framePath);
  if (header.particleCount >
      std::numeric_limits<std::size_t>::max() / kPositionComponents) {
    throw std::overflow_error("N-body frame is too large to decode");
  }

  const std::size_t valueCount =
      static_cast<std::size_t>(header.particleCount) * kPositionComponents;
  std::vector<float> positions(valueCount);
  std::ifstream input(framePath, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open N-body frame: " +
                             framePath.string());
  input.seekg(static_cast<std::streamoff>(header.headerBytes));
  if (!input)
    throw std::runtime_error("Could not seek to N-body frame payload: " +
                             framePath.string());

  if (header.scalarBytes == sizeof(float)) {
    input.read(reinterpret_cast<char *>(positions.data()),
               static_cast<std::streamsize>(valueCount * sizeof(float)));
    if (!input)
      throw std::runtime_error("Failed while reading N-body frame payload: " +
                               framePath.string());
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    for (float &position : positions) {
      std::uint32_t bits = 0;
      std::memcpy(&bits, &position, sizeof(bits));
      bits = ((bits & 0x000000ffU) << 24U) |
             ((bits & 0x0000ff00U) << 8U) |
             ((bits & 0x00ff0000U) >> 8U) |
             ((bits & 0xff000000U) >> 24U);
      std::memcpy(&position, &bits, sizeof(bits));
    }
#endif
    return positions;
  }

  std::vector<std::uint16_t> quantized(
      std::min(kFrameChunkParticles * kPositionComponents, valueCount));
  for (std::size_t offset = 0; offset < valueCount;) {
    const std::size_t count = std::min(quantized.size(), valueCount - offset);
    input.read(reinterpret_cast<char *>(quantized.data()),
               static_cast<std::streamsize>(count * sizeof(std::uint16_t)));
    if (!input)
      throw std::runtime_error("Failed while reading N-body frame payload: " +
                               framePath.string());
    for (std::size_t index = 0; index < count; ++index) {
      std::uint16_t value = quantized[index];
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
      value = static_cast<std::uint16_t>((value << 8U) | (value >> 8U));
#endif
      const std::size_t component = (offset + index) % kPositionComponents;
      const double minimum = header.bounds.minimum[component];
      const double range = header.bounds.maximum[component] - minimum;
      const double decoded =
          range == 0.0
              ? minimum
              : minimum + range * static_cast<double>(value) /
                              static_cast<double>(kQuantizedPositionMaximum);
      positions[offset + index] = static_cast<float>(decoded);
    }
    offset += count;
  }
  return positions;
}

std::vector<std::uint8_t>
readFrameParticleTypes(const std::filesystem::path &framePath) {
  const FrameHeader header = readFrameHeader(framePath);
  if (header.particleCount > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("N-body frame has too many particle types");

  const std::size_t particleCount =
      static_cast<std::size_t>(header.particleCount);
  std::vector<std::uint8_t> types(
      particleCount, static_cast<std::uint8_t>(ParticleType::Unknown));
  if (header.particleTypeBytes == 0)
    return types;

  const std::uint64_t positionBytes =
      header.particleCount * kPositionComponents * header.scalarBytes;
  if (positionBytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) -
          header.headerBytes) {
    throw std::overflow_error("N-body frame type payload offset is too large");
  }

  std::ifstream input(framePath, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Could not open N-body frame: " +
                             framePath.string());
  input.seekg(static_cast<std::streamoff>(header.headerBytes + positionBytes));
  input.read(reinterpret_cast<char *>(types.data()),
             static_cast<std::streamsize>(types.size()));
  if (!input)
    throw std::runtime_error("Failed while reading frame particle types: " +
                             framePath.string());
  for (const std::uint8_t type : types) {
    if (!isValidParticleType(type))
      throw std::runtime_error("N-body frame contains an invalid particle type: " +
                               framePath.string());
  }
  return types;
}

} // namespace nbody::streaming
