#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

import particle;

namespace {

std::filesystem::path makeTemporaryParticleFile()
{
    std::random_device randomDevice;
    const auto token =
        (static_cast<std::uint64_t>(randomDevice()) << 32) |
        static_cast<std::uint64_t>(randomDevice());

    return std::filesystem::temp_directory_path() /
           ("nbody-particles-" + std::to_string(token) + ".bin");
}

} // namespace

class ParticleFileTest : public ::testing::Test {
protected:
    const std::filesystem::path testFile =
        makeTemporaryParticleFile();

    void writePlanarFile(bool use64BitHeader) {
        uint32_t num_particles = 2;
        uint64_t num_particles_64 = num_particles;
        std::vector<double> mass = {1.5e30, 2.5e30};
        std::vector<double> x = {1.0, 2.0};
        std::vector<double> y = {3.0, 4.0};
        std::vector<double> z = {5.0, 6.0};
        std::vector<double> vx = {0.1, 0.2};
        std::vector<double> vy = {0.3, 0.4};
        std::vector<double> vz = {0.5, 0.6};

        std::ofstream outFile(testFile, std::ios::binary);
        if (!outFile) {
            throw std::runtime_error(
                "Could not create temporary particle file");
        }

        if (use64BitHeader) {
            outFile.write(reinterpret_cast<char*>(&num_particles_64), sizeof(uint64_t));
        } else {
            outFile.write(reinterpret_cast<char*>(&num_particles), sizeof(uint32_t));
        }
        outFile.write(reinterpret_cast<char*>(mass.data()), mass.size() * sizeof(double));
        outFile.write(reinterpret_cast<char*>(x.data()), x.size() * sizeof(double));
        outFile.write(reinterpret_cast<char*>(y.data()), y.size() * sizeof(double));
        outFile.write(reinterpret_cast<char*>(z.data()), z.size() * sizeof(double));
        outFile.write(reinterpret_cast<char*>(vx.data()), vx.size() * sizeof(double));
        outFile.write(reinterpret_cast<char*>(vy.data()), vy.size() * sizeof(double));
        outFile.write(reinterpret_cast<char*>(vz.data()), vz.size() * sizeof(double));
    }

    void SetUp() override {
        writePlanarFile(false);
    }

    void TearDown() override {
        std::error_code error;
        std::filesystem::remove(testFile, error);
    }
};

TEST_F(ParticleFileTest, LoadsPlanarFileWith32BitHeader) {
    std::ifstream inFile(testFile, std::ios::binary);
    Particles mw(inFile);

    EXPECT_EQ(mw.num_particles, 2);
    EXPECT_DOUBLE_EQ(mw.mass[0], 1.5e30);
    EXPECT_DOUBLE_EQ(mw.mass[1], 2.5e30);
    EXPECT_DOUBLE_EQ(mw.x[0], 1.0);
    EXPECT_DOUBLE_EQ(mw.x[1], 2.0);
    EXPECT_DOUBLE_EQ(mw.y[0], 3.0);
    EXPECT_DOUBLE_EQ(mw.z[1], 6.0);
    EXPECT_DOUBLE_EQ(mw.vx[0], 0.1);
    EXPECT_DOUBLE_EQ(mw.vy[1], 0.4);
    EXPECT_DOUBLE_EQ(mw.vz[0], 0.5);
}

TEST_F(ParticleFileTest, RejectsUnreadableFile) {
    auto missingFile = testFile;
    missingFile += ".missing";
    std::ifstream inFile(missingFile, std::ios::binary);
    EXPECT_THROW(Particles mw(inFile), std::runtime_error);
}

TEST_F(ParticleFileTest, LoadsPlanarFileWith64BitHeader) {
    writePlanarFile(true);

    std::ifstream inFile(testFile, std::ios::binary);
    Particles mw(inFile);

    EXPECT_EQ(mw.num_particles, 2);
    EXPECT_DOUBLE_EQ(mw.mass[0], 1.5e30);
    EXPECT_DOUBLE_EQ(mw.x[0], 1.0);
    EXPECT_DOUBLE_EQ(mw.y[1], 4.0);
    EXPECT_DOUBLE_EQ(mw.z[0], 5.0);
    EXPECT_DOUBLE_EQ(mw.vx[1], 0.2);
    EXPECT_DOUBLE_EQ(mw.vy[0], 0.3);
    EXPECT_DOUBLE_EQ(mw.vz[1], 0.6);
}
