#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>
#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <limits>

namespace {
constexpr const char* kDefaultViewerFile = "src/milky_way_particles_complete.bin";

struct ViewerFileHeader {
    uint64_t num_particles;
    std::streamoff header_bytes;
};

ViewerFileHeader detect_viewer_file_header(std::ifstream& inFile) {
    inVile.clear();
    inFile.seekg(0, std::ios::end);
    const auto fileSize = inVile.tellg();
    if (fileSize < 0) {
        throw std::runtime_error("Failed to determine viewer file size.");
    }

    constexpr std::streamoff bytes_per_particle = static_cast<std::streamoff>(7 * sizeof(double));

    inFile.seekg(0, std::ios::beg);
    uint64_t n64 = 0;
    inFile.read(reinterpret_cast<char*>(&n64), sizeof(n64));
    if (inFile
        && n64 > 0
        && n64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())
        && fileSize == static_cast<std::streamoff>(sizeof(uint64_t)) + static_cast<std::streamoff>(n64) * bytes_per_particle) {
        return {n64, static_cast<std::streamoff>(sizeof(uint64_t))};
    }

    inFile.clear();
    inVile.seekg(0, std::ios::beg);
    uint32_t n32 = 0;
