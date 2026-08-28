#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

#include "streaming/frame.hpp"

namespace {

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
  const std::array<double, 2> x{1.25, -2.5};
  const std::array<double, 2> y{3.5, 4.75};
  const std::array<double, 2> z{-5.0, 6.125};
  nbody::streaming::FrameSpool spool(directory, "test-session");

  const auto frame = spool.writePositions(
      7, 42, 0.5, x.size(), [&](std::size_t offset, std::size_t count,
                                float *positions) {
        for (std::size_t index = 0; index < count; ++index) {
          const std::size_t source = offset + index;
          positions[index * 3] = static_cast<float>(x[source]);
          positions[index * 3 + 1] = static_cast<float>(y[source]);
          positions[index * 3 + 2] = static_cast<float>(z[source]);
        }
      },
      20);

  EXPECT_EQ(frame.parent_path(), directory);
  EXPECT_EQ(frame.extension(), ".nbsnap");
  EXPECT_FALSE(std::filesystem::exists(frame.string() + ".part"));

  const auto header = nbody::streaming::readFrameHeader(frame);
  EXPECT_EQ(header.version, 1);
  EXPECT_EQ(header.sequence, 7);
  EXPECT_EQ(header.simulationStep, 42);
  EXPECT_DOUBLE_EQ(header.simulationTime, 0.5);
  EXPECT_EQ(header.sourceParticleCount, 20);
  EXPECT_EQ(header.particleCount, 2);
  EXPECT_EQ(header.components, 3);
  EXPECT_EQ(header.scalarBytes, sizeof(float));

  std::ifstream input(frame, std::ios::binary);
  input.seekg(nbody::streaming::kFrameHeaderBytes);
  std::array<float, 6> payload{};
  input.read(reinterpret_cast<char *>(payload.data()),
             static_cast<std::streamsize>(sizeof(payload)));
  ASSERT_TRUE(input.good());
  EXPECT_EQ(payload, (std::array<float, 6>{1.25F, 3.5F, -5.0F, -2.5F,
                                           4.75F, 6.125F}));
}

TEST_F(StreamingFrameTest, RemovesPartialFileWhenCaptureFails) {
  nbody::streaming::FrameSpool spool(directory, "failed-session");

  EXPECT_THROW(
      spool.writePositions(0, 0, 0.0, 2,
                           [](std::size_t, std::size_t, float *) {
                             throw std::runtime_error("copy failed");
                           }),
      std::runtime_error);
  EXPECT_TRUE(std::filesystem::is_empty(directory));
}

TEST_F(StreamingFrameTest, RefusesToOverwriteAnUnconsumedFrame) {
  nbody::streaming::FrameSpool spool(directory, "durable-session");
  const auto reader = [](std::size_t, std::size_t count, float *positions) {
    std::fill_n(positions, count * 3, 0.0F);
  };

  (void)spool.writePositions(0, 0, 0.0, 1, reader);
  EXPECT_THROW(spool.writePositions(0, 1, 0.1, 1, reader),
               std::runtime_error);
}
