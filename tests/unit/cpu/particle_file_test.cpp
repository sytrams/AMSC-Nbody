#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include "src/particle.hpp"

namespace {

enum class HeaderWidth {
  Bits32,
  Bits64,
};

struct ParticleData {
  std::vector<double> mass{1.5e30, 2.5e30};
  std::vector<double> x{1.0, 2.0};
  std::vector<double> y{3.0, 4.0};
  std::vector<double> z{5.0, 6.0};
  std::vector<double> vx{0.1, 0.2};
  std::vector<double> vy{0.3, 0.4};
  std::vector<double> vz{0.5, 0.6};

  std::size_t size() const { return mass.size(); }
};

std::filesystem::path makeTemporaryParticleFile() {
  std::random_device randomDevice;
  const auto token = (static_cast<std::uint64_t>(randomDevice()) << 32) |
                     static_cast<std::uint64_t>(randomDevice());

  return std::filesystem::temp_directory_path() /
         ("nbody-particles-" + std::to_string(token) + ".bin");
}

void expectRuntimeErrorContaining(std::ifstream &input,
                                  const std::string &expectedMessage) {
  try {
    Particles particles(input);
    (void)particles;
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error &error) {
    EXPECT_NE(std::string(error.what()).find(expectedMessage),
              std::string::npos)
        << "Actual message: " << error.what();
  } catch (...) {
    FAIL() << "Expected std::runtime_error";
  }
}

std::vector<double> copyToHost(const double *source, std::size_t count) {
  std::vector<double> result(count);
  thrust::copy_n(thrust::device_pointer_cast(source), count, result.begin());
  return result;
}

void expectDeviceBlockEquals(const std::vector<double> &expected,
                             const double *actual) {
  ASSERT_NE(actual, nullptr);
  const auto hostValues = copyToHost(actual, expected.size());
  ASSERT_EQ(hostValues.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_DOUBLE_EQ(hostValues[index], expected[index]) << "index " << index;
  }
}

void expectViewEquals(const ParticleData &expected,
                      const DeviceParticlesView &actual) {
  EXPECT_EQ(actual.count, expected.size());
  expectDeviceBlockEquals(expected.mass, actual.mass);
  expectDeviceBlockEquals(expected.x, actual.x);
  expectDeviceBlockEquals(expected.y, actual.y);
  expectDeviceBlockEquals(expected.z, actual.z);
  expectDeviceBlockEquals(expected.vx, actual.vx);
  expectDeviceBlockEquals(expected.vy, actual.vy);
  expectDeviceBlockEquals(expected.vz, actual.vz);
}

} // namespace

class ParticleFileTest : public ::testing::Test {
protected:
  const std::filesystem::path testFile = makeTemporaryParticleFile();
  const ParticleData particleData;

  void writeHeader(std::ofstream &output, HeaderWidth width,
                   std::uint64_t count) {
    if (width == HeaderWidth::Bits64) {
      output.write(reinterpret_cast<const char *>(&count),
                   static_cast<std::streamsize>(sizeof(count)));
      return;
    }

    const auto count32 = static_cast<std::uint32_t>(count);
    output.write(reinterpret_cast<const char *>(&count32),
                 static_cast<std::streamsize>(sizeof(count32)));
  }

  void writeBlock(std::ofstream &output, const std::vector<double> &values) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(double)));
  }

  void writePlanarFile(HeaderWidth width) {
    std::ofstream output(testFile, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.is_open());

    writeHeader(output, width, particleData.size());
    writeBlock(output, particleData.mass);
    writeBlock(output, particleData.x);
    writeBlock(output, particleData.y);
    writeBlock(output, particleData.z);
    writeBlock(output, particleData.vx);
    writeBlock(output, particleData.vy);
    writeBlock(output, particleData.vz);
    ASSERT_TRUE(output.good());
  }

  void writeHeaderOnly(HeaderWidth width, std::uint64_t count) {
    std::ofstream output(testFile, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(output.is_open());
    writeHeader(output, width, count);
    ASSERT_TRUE(output.good());
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove(testFile, error);
  }
};

TEST(ParticleConstantsTest, ExposesPhysicalConstants) {
  EXPECT_DOUBLE_EQ(G, 6.67430e-11);
  EXPECT_DOUBLE_EQ(K, 8.98755e9);
}

TEST_F(ParticleFileTest, LoadsEveryBlockWith32BitHeader) {
  writePlanarFile(HeaderWidth::Bits32);
  std::ifstream input(testFile, std::ios::binary);

  Particles particles(input);

  EXPECT_EQ(particles.size(), particleData.size());
  expectViewEquals(particleData, particles.device_view());
}

TEST_F(ParticleFileTest, LoadsEveryBlockWith64BitHeader) {
  writePlanarFile(HeaderWidth::Bits64);
  std::ifstream input(testFile, std::ios::binary);

  Particles particles(input);

  EXPECT_EQ(particles.size(), particleData.size());
  expectViewEquals(particleData, particles.device_view());
}

TEST_F(ParticleFileTest, LoadsRegardlessOfInitialStreamPosition) {
  writePlanarFile(HeaderWidth::Bits32);
  std::ifstream input(testFile, std::ios::binary);
  input.seekg(0, std::ios::end);
  ASSERT_TRUE(input.good());

  Particles particles(input);

  expectViewEquals(particleData, particles.device_view());
}

TEST_F(ParticleFileTest, DeviceViewProvidesWritableParticleStorage) {
  writePlanarFile(HeaderWidth::Bits32);
  std::ifstream input(testFile, std::ios::binary);
  Particles particles(input);
  auto view = particles.device_view();
  const double replacement = 42.5;

  thrust::copy_n(&replacement, 1, thrust::device_pointer_cast(view.x));

  const auto updatedX = copyToHost(particles.device_view().x, particles.size());
  ASSERT_EQ(updatedX.size(), particleData.size());
  EXPECT_DOUBLE_EQ(updatedX.front(), replacement);
  EXPECT_DOUBLE_EQ(updatedX.back(), particleData.x.back());
}

TEST_F(ParticleFileTest, RejectsUnopenedStream) {
  std::ifstream input;
  expectRuntimeErrorContaining(input, "Invalid or unreadable file stream");
}

TEST_F(ParticleFileTest, RejectsStreamWithErrorState) {
  writePlanarFile(HeaderWidth::Bits32);
  std::ifstream input(testFile, std::ios::binary);
  input.setstate(std::ios::badbit);

  expectRuntimeErrorContaining(input, "Invalid or unreadable file stream");
}

TEST_F(ParticleFileTest, RejectsEmptyFile) {
  std::ofstream output(testFile, std::ios::binary | std::ios::trunc);
  output.close();
  std::ifstream input(testFile, std::ios::binary);

  expectRuntimeErrorContaining(input, "Unsupported particle file layout");
}

TEST_F(ParticleFileTest, RejectsZeroParticleCount) {
  writeHeaderOnly(HeaderWidth::Bits32, 0);
  std::ifstream input(testFile, std::ios::binary);

  expectRuntimeErrorContaining(input, "Unsupported particle file layout");
}

TEST_F(ParticleFileTest, RejectsTruncatedParticleData) {
  writePlanarFile(HeaderWidth::Bits64);
  const auto completeSize = std::filesystem::file_size(testFile);
  ASSERT_GT(completeSize, sizeof(double));
  std::filesystem::resize_file(testFile, completeSize - sizeof(double));
  std::ifstream input(testFile, std::ios::binary);

  expectRuntimeErrorContaining(input, "Unsupported particle file layout");
}

TEST_F(ParticleFileTest, RejectsTrailingParticleData) {
  writePlanarFile(HeaderWidth::Bits32);
  std::ofstream output(testFile, std::ios::binary | std::ios::app);
  const std::array<char, 1> trailingByte{0};
  output.write(trailingByte.data(), trailingByte.size());
  output.close();
  std::ifstream input(testFile, std::ios::binary);

  expectRuntimeErrorContaining(input, "Unsupported particle file layout");
}

TEST_F(ParticleFileTest, Rejects64BitCountBeyondSupportedRange) {
  const auto unsupportedCount =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  writeHeaderOnly(HeaderWidth::Bits64, unsupportedCount);
  std::ifstream input(testFile, std::ios::binary);

  expectRuntimeErrorContaining(input, "Unsupported particle file layout");
}
