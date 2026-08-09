#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "particle.hpp"

class MilkyWayTest : public ::testing::Test {
protected:
    const std::string test_file = "test_mw_tmp.bin";

    void write_planar_file(bool use_64_bit_header) {
        uint32_t num_particles = 2;
        uint64_t num_particles_64 = num_particles;
        std::vector<double> mass = {1.5e30, 2.5e30};
        std::vector<double> x = {1.0, 2.0};
        std::vector<double> y = {3.0, 4.0};
        std::vector<double> z = {5.0, 6.0};
        std::vector<double> vx = {0.1, 0.2};
        std::vector<double> vy = {0.3, 0.4};
        std::vector<double> vz = {0.5, 0.6};

        std::ofstream outFile(test_file, std::ios::binary);
        if (use_64_bit_header) {
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
        write_planar_file(false);
    }

    void TearDown() override {
        if (std::filesystem::exists(test_file)) {
            std::filesystem::remove(test_file);
        }
    }
};

TEST_F(MilkyWayTest, LoadDataCorrectly) {
    std::ifstream inFile(test_file, std::ios::binary);
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

TEST_F(MilkyWayTest, ThrowsOnInvalidFile) {
    std::ifstream inFile("non_existent_mw.bin", std::ios::binary);
    EXPECT_THROW(Particles mw(inFile), std::runtime_error);
}

TEST_F(MilkyWayTest, LoadDataCorrectlyWithUInt64Header) {
    write_planar_file(true);

    std::ifstream inFile(test_file, std::ios::binary);
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
