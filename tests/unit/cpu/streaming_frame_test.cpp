#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "streaming/frame.hpp"
#include "particle_type.hpp"

namespace {

template <typename Unsigned>
void writeLittleEndian(std::ostream &output, Unsigned value) {
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    output.put(static_cast<char>((value >> (byte * 8U)) & 0xffU));
}

void writeDouble(std::ostream &output, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  writeLittleEndian(output, bits);
}

void writeLegacyFrame(const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary);
  output.write("NBSNAP01", 8);
  writeLittleEndian(output, std::uint32_t{1});
  writeLittleEndian(output, std::uint32_t{72});
  writeLittleEndian(output, std::uint64_t{3});
  writeLittleEndian(output, std::uint64_t{9});
  writeDouble(output, 0.25);
  writeLittleEndian(output, std::uint64_t{1});
  writeLittleEndian(output, std::uint64_t{1});
  writeLittleEndian(output, std::uint32_t{4});
  writeLittleEndian(output, std::uint32_t{3});
  writeLittleEndian(output, std::uint64_t{12});
  for (const float position : std::array<float, 3>{1.5F, -2.0F, 4.25F}) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &position, sizeof(bits));
    writeLittleEndian(output, bits);
  }
}

std::filesystem::path makeTemporaryDirectory() {
  std::random_device randomDevice;
  const auto token = (static_cast<std::uint64_t>(randomDevice()) << 32U) |
                     static_cast<std::uint64_t>(randomDevice());
  return std::filesystem::temp_directory_path() /
         ("nbody-stream-test-" + std::to_string(token));
}

class StreamingFrameTest : public ::testing::Test {
protected:
  const std::filesystem::path directory = makeTemporaryDirectory();

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

} // namespace

TEST(FrameSamplerTest, SamplesInitialStateAtSixtyWallClockHertz) {
  nbody::streaming::FrameSampler sampler(60.0);

  EXPECT_TRUE(sampler.isDue(0.0));
  EXPECT_EQ(sampler.takeSequence(), 0);
  EXPECT_FALSE(sampler.isDue(0.01));
  EXPECT_TRUE(sampler.isDue(1.0 / 60.0));
  EXPECT_EQ(sampler.takeSequence(), 1);
  EXPECT_FALSE(sampler.isDue(0.02));
  EXPECT_TRUE(sampler.isDue(2.0 / 60.0));
  EXPECT_EQ(sampler.takeSequence(), 2);
  EXPECT_DOUBLE_EQ(sampler.rate(), 60.0);
}

TEST(FrameSamplerTest, EmitsOnlyOneFrameWhenATimestepSkipsIntervals) {
  nbody::streaming::FrameSampler sampler(60.0);

  EXPECT_TRUE(sampler.isDue(0.0));
  EXPECT_TRUE(sampler.isDue(10.0));
  EXPECT_FALSE(sampler.isDue(10.0));
}

TEST(FrameSamplerTest, RejectsInvalidRateAndTime) {
  EXPECT_THROW(nbody::streaming::FrameSampler(0.0), std::invalid_argument);
  nbody::streaming::FrameSampler sampler;
  EXPECT_THROW((void)sampler.isDue(-1.0), std::invalid_argument);
}

TEST_F(StreamingFrameTest, AtomicallyWritesSelfDescribingPositionFrame) {
  const nbody::streaming::PositionBounds bounds{
      {-2.5, 3.5, -5.0}, {1.25, 4.75, 6.125}};
  const std::array<std::uint16_t, 6> quantized{
      65535, 0, 0, 0, 65535, 65535};
  const std::array<std::uint8_t, 2> types{
      static_cast<std::uint8_t>(ParticleType::Star),
      static_cast<std::uint8_t>(ParticleType::Asteroid)};
  nbody::streaming::FrameSpool spool(directory, "test-session");

  const auto frame = spool.writeQuantizedPositions(
      7, 42, 0.5, 2, bounds,
      [&](std::size_t offset, std::size_t count, std::uint16_t *positions) {
        std::copy_n(quantized.data() + offset * 3, count * 3, positions);
      },
      20,
      [&](std::size_t offset, std::size_t count, std::uint8_t *outputTypes) {
        std::copy_n(types.data() + offset, count, outputTypes);
      });

  EXPECT_EQ(frame.parent_path(), directory);
  EXPECT_EQ(frame.extension(), ".nbsnap");
  EXPECT_FALSE(std::filesystem::exists(frame.string() + ".part"));
  EXPECT_EQ(std::filesystem::file_size(frame),
            nbody::streaming::kFrameHeaderBytes + quantized.size() * 2 +
                types.size());

  const auto header = nbody::streaming::readFrameHeader(frame);
  EXPECT_EQ(header.version, nbody::streaming::kFrameVersion);
  EXPECT_EQ(header.headerBytes, nbody::streaming::kFrameHeaderBytes);
  EXPECT_EQ(header.sequence, 7);
  EXPECT_EQ(header.simulationStep, 42);
  EXPECT_DOUBLE_EQ(header.simulationTime, 0.5);
  EXPECT_EQ(header.sourceParticleCount, 20);
  EXPECT_EQ(header.particleCount, 2);
  EXPECT_EQ(header.components, 3);
  EXPECT_EQ(header.scalarBytes, sizeof(std::uint16_t));
  EXPECT_EQ(header.payloadBytes, 14);
  EXPECT_EQ(header.particleTypeBytes, 2);
  EXPECT_EQ(header.bounds.minimum, bounds.minimum);
  EXPECT_EQ(header.bounds.maximum, bounds.maximum);

  std::ifstream input(frame, std::ios::binary);
  input.seekg(nbody::streaming::kFrameHeaderBytes);
  std::array<std::uint16_t, 6> payload{};
  input.read(reinterpret_cast<char *>(payload.data()),
             static_cast<std::streamsize>(sizeof(payload)));
  ASSERT_TRUE(input.good());
  EXPECT_EQ(payload, quantized);

  EXPECT_EQ(nbody::streaming::readFramePositions(frame),
            (std::vector<float>{1.25F, 3.5F, -5.0F, -2.5F, 4.75F,
                                6.125F}));
  EXPECT_EQ(nbody::streaming::readFrameParticleTypes(frame),
            (std::vector<std::uint8_t>{types[0], types[1]}));
}

TEST_F(StreamingFrameTest, RemovesPartialFileWhenCaptureFails) {
  nbody::streaming::FrameSpool spool(directory, "failed-session");
  const nbody::streaming::PositionBounds bounds{{0.0, 0.0, 0.0},
                                                 {1.0, 1.0, 1.0}};

  EXPECT_THROW(
      spool.writeQuantizedPositions(
          0, 0, 0.0, 2, bounds,
          [](std::size_t, std::size_t, std::uint16_t *) {
            throw std::runtime_error("copy failed");
          }),
      std::runtime_error);
  EXPECT_TRUE(std::filesystem::is_empty(directory));
}

TEST_F(StreamingFrameTest, RefusesToOverwriteAnUnconsumedFrame) {
  nbody::streaming::FrameSpool spool(directory, "durable-session");
  const nbody::streaming::PositionBounds bounds{{0.0, 0.0, 0.0},
                                                 {0.0, 0.0, 0.0}};
  const auto reader = [](std::size_t, std::size_t count,
                         std::uint16_t *positions) {
    std::fill_n(positions, count * 3, std::uint16_t{0});
  };

  (void)spool.writeQuantizedPositions(0, 0, 0.0, 1, bounds, reader);
  EXPECT_THROW(
      spool.writeQuantizedPositions(0, 1, 0.1, 1, bounds, reader),
      std::runtime_error);
}

TEST_F(StreamingFrameTest, DecodesCollapsedQuantizationAxes) {
  nbody::streaming::FrameSpool spool(directory, "collapsed-axis");
  const nbody::streaming::PositionBounds bounds{{8.0, -1.0, 5.0},
                                                 {8.0, 3.0, 5.0}};
  const auto frame = spool.writeQuantizedPositions(
      0, 0, 0.0, 1, bounds,
      [](std::size_t, std::size_t, std::uint16_t *positions) {
        positions[0] = 65535;
        positions[1] = 32768;
        positions[2] = 12345;
      });

  const auto positions = nbody::streaming::readFramePositions(frame);
  ASSERT_EQ(positions.size(), 3);
  EXPECT_FLOAT_EQ(positions[0], 8.0F);
  EXPECT_NEAR(positions[1], 1.0F, 4.0F / 65535.0F);
  EXPECT_FLOAT_EQ(positions[2], 5.0F);
  EXPECT_EQ(nbody::streaming::readFrameParticleTypes(frame),
            (std::vector<std::uint8_t>{
                static_cast<std::uint8_t>(ParticleType::Unknown)}));
}

TEST_F(StreamingFrameTest, ReadsLegacyFloatPositionFrame) {
  std::filesystem::create_directories(directory);
  const auto frame = directory / "legacy.nbsnap";
  writeLegacyFrame(frame);

  const auto header = nbody::streaming::readFrameHeader(frame);
  EXPECT_EQ(header.version, nbody::streaming::kLegacyFrameVersion);
  EXPECT_EQ(header.headerBytes, nbody::streaming::kLegacyFrameHeaderBytes);
  EXPECT_EQ(header.scalarBytes, sizeof(float));
  EXPECT_EQ(nbody::streaming::readFramePositions(frame),
            (std::vector<float>{1.5F, -2.0F, 4.25F}));
  EXPECT_EQ(nbody::streaming::readFrameParticleTypes(frame),
            (std::vector<std::uint8_t>{
                static_cast<std::uint8_t>(ParticleType::Unknown)}));
}

TEST_F(StreamingFrameTest, PublishesCompletionMarkerAtomically) {
  nbody::streaming::FrameSpool spool(directory, "completed-session");
  const auto marker = spool.markComplete();

  EXPECT_EQ(marker.filename(), "run-completed-session.complete");
  EXPECT_TRUE(std::filesystem::is_regular_file(marker));
  EXPECT_FALSE(std::filesystem::exists(marker.string() + ".part"));
  EXPECT_EQ(spool.markComplete(), marker);
}

TEST_F(StreamingFrameTest, RejectsInvalidQuantizationBounds) {
  nbody::streaming::FrameSpool spool(directory, "invalid-bounds");
  const nbody::streaming::PositionBounds bounds{{1.0, 0.0, 0.0},
                                                 {-1.0, 0.0, 0.0}};
  EXPECT_THROW(
      spool.writeQuantizedPositions(
          0, 0, 0.0, 1, bounds,
          [](std::size_t, std::size_t, std::uint16_t *) {}),
      std::invalid_argument);
}
