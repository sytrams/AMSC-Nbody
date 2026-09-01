#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "particle_type.hpp"
#include "streaming/frame.hpp"

#ifndef NBODY_VIEWER_SHADER_PATH
#define NBODY_VIEWER_SHADER_PATH "apps/viewer/renderer.metal"
#endif

namespace {
namespace fs = std::filesystem;
using nbody::streaming::FrameHeader;

constexpr char kLatestFrameMarker[] = ".nbody-latest";
constexpr char kStreamStateMarker[] = ".nbody-stream-state";
constexpr char kDefaultViewerFile[] = "data/test_spiral.bin";

struct ParticleData {
  simd_float3 position;
  std::uint32_t type = static_cast<std::uint32_t>(ParticleType::Unknown);
};
static_assert(sizeof(ParticleData) == 32,
              "Metal particle vertices must have a 32-byte stride");

struct ViewerUniforms {
  simd_float4x4 viewProjection;
  simd_float4 display;
};

struct Snapshot {
  FrameHeader header;
  std::vector<ParticleData> particles;
  fs::path path;
  bool streamed = true;
};

struct ViewerOptions {
  std::optional<fs::path> file;
  std::optional<fs::path> streamDirectory;
  std::optional<fs::path> replayDirectory;
  double pollMilliseconds = 16.0;
  double replayFramesPerSecond = 30.0;
  bool validateOnly = false;
};

struct StreamState {
  std::string generation;
  bool ended = false;
};

std::string_view requireValue(int argc, char **argv, int &index,
                              std::string_view option) {
  if (index + 1 >= argc)
    throw std::invalid_argument("Missing value for " + std::string(option));
  return argv[++index];
}

double parsePositiveDouble(std::string_view text, std::string_view option) {
  double value = 0.0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      !std::isfinite(value) || value <= 0.0)
    throw std::invalid_argument("Invalid value for " + std::string(option) +
                                ": " + std::string(text));
  return value;
}

void printUsage(std::ostream &output, const char *program) {
  output
      << "Usage: " << program << " [OPTIONS]\n\n"
      << "  --stream-dir DIR  Follow complete .nbsnap files downloaded to DIR\n"
      << "  --replay-dir DIR  Loop an archived directory of .nbsnap files\n"
      << "  -f, --file FILE   Display one .nbsnap or planar particle file\n"
      << "  --poll-ms MS      Live marker polling interval (default: 16)\n"
      << "  --replay-fps FPS   Loop completed streams at this rate (default: 30)\n"
      << "  --validate-only   Load one frame, print its metadata, and exit\n"
      << "  -h, --help        Show this help and exit\n\n"
      << "Controls: drag to rotate, two-finger scroll or arrows to pan, pinch "
         "or +/- to zoom, click a particle to follow it, Esc to release, F "
         "to fit, R to reset rotation.\n";
}

ViewerOptions parseOptions(int argc, char **argv) {
  ViewerOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "-h" || option == "--help") {
      printUsage(std::cout, argv[0]);
      std::exit(0);
    } else if (option == "-f" || option == "--file") {
      options.file = requireValue(argc, argv, index, option);
    } else if (option.starts_with("--file=")) {
      options.file = std::string(option.substr(7));
    } else if (option == "--stream-dir") {
      options.streamDirectory = requireValue(argc, argv, index, option);
    } else if (option.starts_with("--stream-dir=")) {
      options.streamDirectory = std::string(option.substr(13));
    } else if (option == "--replay-dir") {
      options.replayDirectory = requireValue(argc, argv, index, option);
    } else if (option.starts_with("--replay-dir=")) {
      options.replayDirectory = std::string(option.substr(13));
    } else if (option == "--poll-ms") {
      options.pollMilliseconds = parsePositiveDouble(
          requireValue(argc, argv, index, option), option);
    } else if (option == "--replay-fps") {
      options.replayFramesPerSecond = parsePositiveDouble(
          requireValue(argc, argv, index, option), option);
    } else if (option == "--validate-only") {
      options.validateOnly = true;
    } else {
      throw std::invalid_argument("Unknown option: " + std::string(option));
    }
  }
  const int inputModes = static_cast<int>(options.file.has_value()) +
                         static_cast<int>(options.streamDirectory.has_value()) +
                         static_cast<int>(options.replayDirectory.has_value());
  if (inputModes > 1)
    throw std::invalid_argument(
        "--file, --stream-dir, and --replay-dir cannot be used together");
  if (inputModes == 0)
    options.file = fs::path(kDefaultViewerFile);
  return options;
}

std::size_t renderSlot(std::uint32_t type) {
  switch (static_cast<ParticleType>(type)) {
  case ParticleType::Asteroid:
    return 0;
  case ParticleType::Unknown:
    return 1;
  case ParticleType::Moon:
    return 2;
  case ParticleType::Planet:
    return 3;
  case ParticleType::Star:
    return 4;
  }
  return 1;
}

std::vector<ParticleData>
unpackPositions(const std::vector<float> &values,
                const std::vector<std::uint8_t> &types) {
  std::vector<ParticleData> particles(values.size() / 3);
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const float x = values[index * 3];
    const float y = values[index * 3 + 1];
    const float z = values[index * 3 + 2];
    particles[index].position = {
        std::isfinite(x) ? x : 0.0F,
        std::isfinite(y) ? y : 0.0F,
        std::isfinite(z) ? z : 0.0F,
    };
    particles[index].type = types[index];
  }

  // Metal blends point primitives in submission order. A stable counting
  // sort renders the dense asteroid cloud first and the rare large bodies
  // last, keeping planets and the Sun visible without changing identity
  // between frames.
  std::array<std::size_t, 5> counts{};
  for (const ParticleData &particle : particles)
    ++counts[renderSlot(particle.type)];
  std::array<std::size_t, 5> offsets{};
  for (std::size_t slot = 1; slot < offsets.size(); ++slot)
    offsets[slot] = offsets[slot - 1] + counts[slot - 1];
  std::array<std::size_t, 5> next = offsets;
  std::vector<ParticleData> sorted(particles.size());
  for (const ParticleData &particle : particles)
    sorted[next[renderSlot(particle.type)]++] = particle;
  return sorted;
}

Snapshot loadStreamSnapshot(const fs::path &path) {
  const FrameHeader header = nbody::streaming::readFrameHeader(path);
  const std::vector<float> values =
      nbody::streaming::readFramePositions(path);
  const std::vector<std::uint8_t> types =
      nbody::streaming::readFrameParticleTypes(path);
  return {header, unpackPositions(values, types), path, true};
}

struct PlanarHeader {
  std::uint64_t particleCount;
  std::streamoff headerBytes;
  bool hasParticleTypes;
};

PlanarHeader detectPlanarHeader(std::ifstream &input) {
  input.clear();
  input.seekg(0, std::ios::end);
  const auto fileBytes = input.tellg();
  if (fileBytes < 0)
    throw std::runtime_error("Could not determine particle file size");
  constexpr std::streamoff particleBytes = 7 * sizeof(double);

  const auto matchesLayout = [&](std::uint64_t count,
                                 std::streamoff headerBytes,
                                 bool hasParticleTypes) {
    const auto typeBytes = hasParticleTypes
                               ? static_cast<std::streamoff>(count)
                               : std::streamoff{0};
    return fileBytes == headerBytes + static_cast<std::streamoff>(count) *
                                           particleBytes + typeBytes;
  };

  input.seekg(0, std::ios::beg);
  std::uint64_t count64 = 0;
  input.read(reinterpret_cast<char *>(&count64), sizeof(count64));
  if (input && count64 > 0 &&
      count64 <= static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max())) {
    const auto headerBytes = static_cast<std::streamoff>(sizeof(count64));
    if (matchesLayout(count64, headerBytes, true))
      return {count64, headerBytes, true};
    if (matchesLayout(count64, headerBytes, false))
      return {count64, headerBytes, false};
  }

  input.clear();
  input.seekg(0, std::ios::beg);
  std::uint32_t count32 = 0;
  input.read(reinterpret_cast<char *>(&count32), sizeof(count32));
  if (input && count32 > 0) {
    const auto headerBytes = static_cast<std::streamoff>(sizeof(count32));
    if (matchesLayout(count32, headerBytes, true))
      return {count32, headerBytes, true};
    if (matchesLayout(count32, headerBytes, false))
      return {count32, headerBytes, false};
  }
  throw std::runtime_error(
      "Unsupported file layout; expected NBSNAP01 or planar particle data "
      "with an optional particle-type block");
}

Snapshot loadPlanarSnapshot(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Could not open particle file: " + path.string());
  const PlanarHeader planar = detectPlanarHeader(input);
  const std::size_t count = static_cast<std::size_t>(planar.particleCount);
  const auto blockBytes = static_cast<std::streamoff>(count * sizeof(double));
  input.clear();
  input.seekg(planar.headerBytes + blockBytes, std::ios::beg); // Skip mass.

  std::vector<double> x(count), y(count), z(count);
  input.read(reinterpret_cast<char *>(x.data()), blockBytes);
  input.read(reinterpret_cast<char *>(y.data()), blockBytes);
  input.read(reinterpret_cast<char *>(z.data()), blockBytes);
  if (!input)
    throw std::runtime_error("Could not read planar particle positions: " +
                             path.string());

  std::vector<std::uint8_t> types(
      count, static_cast<std::uint8_t>(ParticleType::Unknown));
  if (planar.hasParticleTypes) {
    input.seekg(3 * blockBytes, std::ios::cur); // Skip vx, vy, vz.
    input.read(reinterpret_cast<char *>(types.data()),
               static_cast<std::streamsize>(types.size()));
    if (!input ||
        !std::all_of(types.begin(), types.end(), isValidParticleType))
      throw std::runtime_error("Could not read valid planar particle types: " +
                               path.string());
  }

  std::vector<float> values(count * 3);
  for (std::size_t index = 0; index < count; ++index) {
    values[index * 3] =
        std::isfinite(x[index]) ? static_cast<float>(x[index]) : 0.0F;
    values[index * 3 + 1] =
        std::isfinite(y[index]) ? static_cast<float>(y[index]) : 0.0F;
    values[index * 3 + 2] =
        std::isfinite(z[index]) ? static_cast<float>(z[index]) : 0.0F;
  }

  FrameHeader header;
  header.sourceParticleCount = planar.particleCount;
  header.particleCount = planar.particleCount;
  return {header, unpackPositions(values, types), path, false};
}

Snapshot loadSnapshot(const fs::path &path) {
  return path.extension() == ".nbsnap" ? loadStreamSnapshot(path)
                                        : loadPlanarSnapshot(path);
}

std::optional<fs::path> markerSnapshot(const fs::path &directory) {
  std::ifstream marker(directory / kLatestFrameMarker);
  std::string name;
  if (!marker || !std::getline(marker, name) || name.empty())
    return std::nullopt;
  const fs::path relative(name);
  if (relative != relative.filename() || relative.extension() != ".nbsnap")
    throw std::runtime_error("Invalid latest-frame marker in " +
                             directory.string());
  const fs::path frame = directory / relative;
  return fs::is_regular_file(frame) ? std::optional<fs::path>(frame)
                                    : std::nullopt;
}

std::optional<fs::path> newestSnapshot(const fs::path &directory) {
  std::error_code error;
  fs::directory_iterator entries(directory, error);
  if (error)
    throw std::runtime_error("Could not inspect stream directory '" +
                             directory.string() + "': " + error.message());
  std::optional<fs::path> newest;
  fs::file_time_type newestTime{};
  for (const auto &entry : entries) {
    if (!entry.is_regular_file(error) || error ||
        entry.path().extension() != ".nbsnap") {
      error.clear();
      continue;
    }
    const auto currentTime = entry.last_write_time(error);
    if (error) {
      error.clear();
      continue;
    }
    if (!newest || currentTime > newestTime ||
        (currentTime == newestTime && entry.path() > *newest)) {
      newest = entry.path();
      newestTime = currentTime;
    }
  }
  return newest;
}

std::optional<StreamState> readStreamState(const fs::path &directory) {
  std::ifstream input(directory / kStreamStateMarker);
  std::string status;
  StreamState state;
  if (!input || !(input >> status >> state.generation))
    return std::nullopt;
  if (status != "active" && status != "ended")
    throw std::runtime_error("Invalid stream state in " +
                             directory.string());
  state.ended = status == "ended";
  return state;
}

std::optional<std::string> frameSession(const fs::path &path) {
  const std::string name = path.filename().string();
  const std::size_t separator = name.rfind("-frame-");
  if (separator == std::string::npos || separator == 0)
    return std::nullopt;
  return name.substr(0, separator);
}

std::vector<fs::path> sessionSnapshots(const fs::path &directory,
                                       std::string_view session) {
  const std::string prefix = std::string(session) + "-frame-";
  std::vector<fs::path> frames;
  std::error_code error;
  fs::directory_iterator entries(directory, error);
  if (error)
    throw std::runtime_error("Could not inspect stream directory '" +
                             directory.string() + "': " + error.message());
  for (const auto &entry : entries) {
    const std::string name = entry.path().filename().string();
    if (entry.is_regular_file(error) && !error &&
        entry.path().extension() == ".nbsnap" && name.starts_with(prefix))
      frames.push_back(entry.path());
    error.clear();
  }
  std::sort(frames.begin(), frames.end());
  return frames;
}

simd_float4x4 translationMatrix(simd_float3 translation) {
  return simd_matrix(simd_make_float4(1, 0, 0, 0),
                     simd_make_float4(0, 1, 0, 0),
                     simd_make_float4(0, 0, 1, 0),
                     simd_make_float4(translation.x, translation.y,
                                      translation.z, 1));
}

simd_float4x4 rotationX(float angle) {
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  return simd_matrix(simd_make_float4(1, 0, 0, 0),
                     simd_make_float4(0, cosine, sine, 0),
                     simd_make_float4(0, -sine, cosine, 0),
                     simd_make_float4(0, 0, 0, 1));
}

simd_float4x4 rotationY(float angle) {
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  return simd_matrix(simd_make_float4(cosine, 0, -sine, 0),
                     simd_make_float4(0, 1, 0, 0),
                     simd_make_float4(sine, 0, cosine, 0),
                     simd_make_float4(0, 0, 0, 1));
}

NSString *snapshotTitle(const Snapshot &snapshot, bool replay = false) {
  if (!snapshot.streamed)
    return [NSString
        stringWithFormat:@"N-Body Metal Viewer — %llu particles",
                         static_cast<unsigned long long>(
                             snapshot.header.particleCount)];
  const auto currentStep =
      static_cast<unsigned long long>(snapshot.header.simulationStep);
  const auto totalSteps =
      static_cast<unsigned long long>(snapshot.header.totalSteps);
  NSString *stepText = snapshot.header.totalSteps > 0
                           ? [NSString stringWithFormat:@"%llu/%llu", currentStep,
                                                        totalSteps]
                           : [NSString stringWithFormat:@"%llu/?", currentStep];
  if (snapshot.header.sourceParticleCount == snapshot.header.particleCount)
    return [NSString
        stringWithFormat:@"N-Body %@ — step %@ — t %.6g — %llu particles",
                         replay ? @"Replay" : @"Live",
                         stepText,
                         snapshot.header.simulationTime,
                         static_cast<unsigned long long>(
                             snapshot.header.particleCount)];
  return [NSString
      stringWithFormat:
          @"N-Body %@ — step %@ — t %.6g — %llu / %llu particles",
          replay ? @"Replay" : @"Live",
          stepText,
          snapshot.header.simulationTime,
          static_cast<unsigned long long>(snapshot.header.particleCount),
          static_cast<unsigned long long>(snapshot.header.sourceParticleCount)];
}
} // namespace

@interface MetalRenderer : NSObject <MTKViewDelegate>
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                  error:(NSError **)error;
- (void)displaySnapshot:(const Snapshot &)snapshot;
- (void)selectParticleAt:(simd_float2)anchor viewSize:(CGSize)viewSize;
- (void)clearSelection;
- (void)handleKeyDown:(NSEvent *)event;
- (void)handleScroll:(NSEvent *)event
             viewHeight:(CGFloat)viewHeight;
- (void)handleMagnify:(NSEvent *)event
                anchor:(simd_float2)anchor
                aspect:(float)aspect;
- (void)handleDragX:(CGFloat)deltaX
             deltaY:(CGFloat)deltaY;
@end

@implementation MetalRenderer {
  id<MTLDevice> _device;
  id<MTLRenderPipelineState> _pipelineState;
  id<MTLBuffer> _vertexBuffer;
  id<MTLCommandQueue> _commandQueue;
  NSUInteger _particleCount;
  NSUInteger _selectedParticleIndex;
  std::vector<ParticleData> _particles;
  simd_float3 _dataCenter;
  float _dataExtent;
  simd_float3 _viewCenter;
  float _viewExtent;
  simd_float2 _panOffset;
  float _zoomFactor;
  float _yaw;
  float _pitch;
  CGSize _viewSize;
  BOOL _autoFit;
}

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                  error:(NSError **)error {
  self = [super init];
  if (!self)
    return nil;
  _device = device;
  _commandQueue = [device newCommandQueue];
  _vertexBuffer = [device newBufferWithLength:sizeof(ParticleData)
                                       options:MTLResourceStorageModeShared];
  _viewExtent = 1.0F;
  _dataExtent = 1.0F;
  _zoomFactor = 1.0F;
  _selectedParticleIndex = NSNotFound;
  _autoFit = YES;

  NSString *shaderSource =
      [NSString stringWithContentsOfFile:@NBODY_VIEWER_SHADER_PATH
                                encoding:NSUTF8StringEncoding
                                   error:error];
  if (!shaderSource)
    return nil;
  id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
                                                options:nil
                                                  error:error];
  if (!library)
    return nil;

  MTLRenderPipelineDescriptor *descriptor =
      [[MTLRenderPipelineDescriptor alloc] init];
  descriptor.vertexFunction = [library newFunctionWithName:@"vertex_main"];
  descriptor.fragmentFunction = [library newFunctionWithName:@"fragment_main"];
  descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
  descriptor.colorAttachments[0].blendingEnabled = YES;
  descriptor.colorAttachments[0].sourceRGBBlendFactor =
      MTLBlendFactorSourceAlpha;
  descriptor.colorAttachments[0].sourceAlphaBlendFactor =
      MTLBlendFactorSourceAlpha;
  descriptor.colorAttachments[0].destinationRGBBlendFactor =
      MTLBlendFactorOneMinusSourceAlpha;
  descriptor.colorAttachments[0].destinationAlphaBlendFactor =
      MTLBlendFactorOneMinusSourceAlpha;

  MTLVertexDescriptor *vertices = [MTLVertexDescriptor vertexDescriptor];
  vertices.attributes[0].format = MTLVertexFormatFloat3;
  vertices.attributes[0].bufferIndex = 0;
  vertices.attributes[1].format = MTLVertexFormatUInt;
  vertices.attributes[1].offset = offsetof(ParticleData, type);
  vertices.attributes[1].bufferIndex = 0;
  vertices.layouts[0].stride = sizeof(ParticleData);
  descriptor.vertexDescriptor = vertices;
  _pipelineState =
      [device newRenderPipelineStateWithDescriptor:descriptor error:error];
  return _pipelineState ? self : nil;
}

- (void)displaySnapshot:(const Snapshot &)snapshot {
  if (snapshot.particles.empty())
    return;
  id<MTLBuffer> buffer =
      [_device newBufferWithBytes:snapshot.particles.data()
                           length:snapshot.particles.size() * sizeof(ParticleData)
                          options:MTLResourceStorageModeShared];
  if (!buffer) {
    NSLog(@"Could not allocate Metal particle buffer");
    return;
  }

  simd_float3 minimum = snapshot.particles.front().position;
  simd_float3 maximum = minimum;
  for (const ParticleData &particle : snapshot.particles) {
    minimum = simd_min(minimum, particle.position);
    maximum = simd_max(maximum, particle.position);
  }
  const simd_float3 dataCenter = (minimum + maximum) * 0.5F;
  const simd_float3 halfRange = (maximum - minimum) * 0.5F;
  const float extent =
      std::max({halfRange.x, halfRange.y, halfRange.z, 1.0e-6F}) * 1.12F;
  @synchronized(self) {
    _dataCenter = dataCenter;
    _dataExtent = extent;
    _particles = snapshot.particles;
    if (_selectedParticleIndex != NSNotFound &&
        _selectedParticleIndex < _particles.size()) {
      // Sample order is deterministic across frames, so the selected slot is
      // a stable body identity for the lifetime of this stream.
      _viewCenter = _particles[_selectedParticleIndex].position;
    } else {
      _selectedParticleIndex = NSNotFound;
    }
    if (_autoFit || _particleCount == 0) {
      _viewCenter = _dataCenter;
      _viewExtent = _dataExtent;
      _panOffset = {0, 0};
      _zoomFactor = 1.0F;
    }
    _vertexBuffer = buffer;
    _particleCount = snapshot.particles.size();
  }
}

- (ViewerUniforms)uniforms {
  const float aspect = _viewSize.height > 0.0
                           ? static_cast<float>(_viewSize.width / _viewSize.height)
                           : 1.0F;
  const float extent = std::max(_viewExtent / _zoomFactor, 1.0e-12F);
  const simd_float4x4 projection = simd_matrix(
      simd_make_float4(1.0F / (extent * aspect), 0, 0, 0),
      simd_make_float4(0, 1.0F / extent, 0, 0),
      simd_make_float4(0, 0, 1.0F / (extent * 2.0F), 0),
      simd_make_float4(0, 0, 0.5F, 1));
  const simd_float4x4 center = translationMatrix(-_viewCenter);
  const simd_float4x4 rotation =
      simd_mul(rotationX(_pitch), rotationY(_yaw));
  const simd_float4x4 pan =
      translationMatrix(simd_make_float3(-_panOffset.x, -_panOffset.y, 0));
  const float count = static_cast<float>(std::max<NSUInteger>(1, _particleCount));
  const float basePointSize =
      std::clamp(8.0F / std::sqrt(std::max(1.0F, count / 800.0F)),
                 1.25F, 7.0F);
  // Point primitives do not acquire an apparent size from the projection.
  // Grow them sub-linearly with camera zoom so detail becomes easier to see
  // without producing excessive fragment overdraw for large particle sets.
  const float zoomPointScale = std::sqrt(std::max(1.0F, _zoomFactor));
  const float pointSize =
      std::clamp(basePointSize * zoomPointScale, basePointSize, 24.0F);
  const float selectedVertex =
      _selectedParticleIndex == NSNotFound
          ? -1.0F
          : static_cast<float>(_selectedParticleIndex);
  return {simd_mul(projection,
                   simd_mul(pan, simd_mul(rotation, center))),
          simd_make_float4(pointSize, selectedVertex, 0, 0)};
}

- (void)selectParticleAt:(simd_float2)anchor viewSize:(CGSize)viewSize {
  @synchronized(self) {
    if (_particles.empty()) {
      _selectedParticleIndex = NSNotFound;
      return;
    }

    const ViewerUniforms camera = [self uniforms];
    const float width =
        static_cast<float>(std::max(1.0, static_cast<double>(viewSize.width)));
    const float height =
        static_cast<float>(std::max(1.0, static_cast<double>(viewSize.height)));
    constexpr float selectionRadius = 18.0F;
    float nearestDistanceSquared = selectionRadius * selectionRadius;
    NSUInteger nearestIndex = NSNotFound;

    for (std::size_t index = 0; index < _particles.size(); ++index) {
      const simd_float3 position = _particles[index].position;
      const simd_float4 clip = simd_mul(
          camera.viewProjection,
          simd_make_float4(position.x, position.y, position.z, 1.0F));
      if (!std::isfinite(clip.x) || !std::isfinite(clip.y) ||
          !std::isfinite(clip.z) || !std::isfinite(clip.w) ||
          std::abs(clip.w) < 1.0e-12F)
        continue;
      const simd_float3 projected = clip.xyz / clip.w;
      if (projected.x < -1.0F || projected.x > 1.0F ||
          projected.y < -1.0F || projected.y > 1.0F ||
          projected.z < 0.0F || projected.z > 1.0F)
        continue;
      const float deltaX = (projected.x - anchor.x) * width * 0.5F;
      const float deltaY = (projected.y - anchor.y) * height * 0.5F;
      const float distanceSquared = deltaX * deltaX + deltaY * deltaY;
      if (distanceSquared < nearestDistanceSquared) {
        nearestDistanceSquared = distanceSquared;
        nearestIndex = static_cast<NSUInteger>(index);
      }
    }

    _selectedParticleIndex = nearestIndex;
    if (nearestIndex == NSNotFound)
      return;

    // Rotate about the selected body and keep it at the position where it was
    // clicked. Future frames update this center using the same sampled slot.
    _viewCenter = _particles[nearestIndex].position;
    const float extent = std::max(_viewExtent / _zoomFactor, 1.0e-12F);
    const float aspect = width / height;
    _panOffset = simd_make_float2(-anchor.x * extent * aspect,
                                  -anchor.y * extent);
    _autoFit = NO;
  }
}

- (void)clearSelection {
  @synchronized(self) {
    _selectedParticleIndex = NSNotFound;
  }
}

- (void)handleKeyDown:(NSEvent *)event {
  @synchronized(self) {
    const float move = _viewExtent * 0.08F / _zoomFactor;
    NSString *characters = event.charactersIgnoringModifiers.lowercaseString;
    if (event.keyCode == 53) {
      _selectedParticleIndex = NSNotFound;
    } else if (event.keyCode == 123) {
      _panOffset.x -= move;
      _autoFit = NO;
    } else if (event.keyCode == 124) {
      _panOffset.x += move;
      _autoFit = NO;
    } else if (event.keyCode == 125) {
      _panOffset.y -= move;
      _autoFit = NO;
    } else if (event.keyCode == 126) {
      _panOffset.y += move;
      _autoFit = NO;
    } else if ([characters isEqualToString:@"+"] ||
               [characters isEqualToString:@"="]) {
      const float oldZoom = _zoomFactor;
      _zoomFactor = std::min(oldZoom * 1.2F, 1.0e6F);
      if (_selectedParticleIndex != NSNotFound)
        _panOffset *= oldZoom / _zoomFactor;
      _autoFit = NO;
    } else if ([characters isEqualToString:@"-"]) {
      const float oldZoom = _zoomFactor;
      _zoomFactor = std::max(oldZoom / 1.2F, 0.02F);
      if (_selectedParticleIndex != NSNotFound)
        _panOffset *= oldZoom / _zoomFactor;
      _autoFit = NO;
    } else if ([characters isEqualToString:@"f"]) {
      _selectedParticleIndex = NSNotFound;
      _viewCenter = _dataCenter;
      _viewExtent = _dataExtent;
      _panOffset = {0, 0};
      _zoomFactor = 1.0F;
      _autoFit = YES;
    } else if ([characters isEqualToString:@"r"]) {
      _yaw = 0.0F;
      _pitch = 0.0F;
    }
  }
}

- (void)zoomBy:(float)multiplier
         anchor:(simd_float2)anchor
         aspect:(float)aspect {
  @synchronized(self) {
    const float oldZoom = _zoomFactor;
    const float newZoom = std::clamp(oldZoom * multiplier, 0.02F, 1.0e6F);
    if (newZoom == oldZoom)
      return;

    const float oldExtent = _viewExtent / oldZoom;
    const float newExtent = _viewExtent / newZoom;
    if (_selectedParticleIndex != NSNotFound) {
      // Once a body is followed, zoom around its screen position even if the
      // trackpad cursor has drifted away from it.
      anchor.x = std::clamp(
          -_panOffset.x / (oldExtent * std::max(aspect, 1.0e-6F)),
          -1.0F, 1.0F);
      anchor.y = std::clamp(-_panOffset.y / oldExtent, -1.0F, 1.0F);
    }
    const float extentChange = oldExtent - newExtent;
    _panOffset.x += anchor.x * std::max(aspect, 1.0e-6F) * extentChange;
    _panOffset.y += anchor.y * extentChange;
    _zoomFactor = newZoom;
    _autoFit = NO;
  }
}

- (void)handleScroll:(NSEvent *)event
           viewHeight:(CGFloat)viewHeight {
  @synchronized(self) {
    const float scale = _viewExtent * 2.0F /
                        std::max(1.0, static_cast<double>(viewHeight)) /
                        _zoomFactor;
    // Move the rendered bodies in the same direction as the two-finger gesture.
    _panOffset.x += static_cast<float>(event.scrollingDeltaX) * scale;
    _panOffset.y += static_cast<float>(event.scrollingDeltaY) * scale;
    _autoFit = NO;
  }
}

- (void)handleMagnify:(NSEvent *)event
                anchor:(simd_float2)anchor
                aspect:(float)aspect {
  const float multiplier =
      std::max(0.01F, 1.0F + static_cast<float>(event.magnification));
  [self zoomBy:multiplier anchor:anchor aspect:aspect];
}

- (void)handleDragX:(CGFloat)deltaX
             deltaY:(CGFloat)deltaY {
  @synchronized(self) {
    _yaw += static_cast<float>(deltaX) * 0.008F;
    _pitch = std::clamp(_pitch + static_cast<float>(deltaY) * 0.008F,
                        -1.55F, 1.55F);
    _autoFit = NO;
  }
}

- (void)drawInMTKView:(MTKView *)view {
  MTLRenderPassDescriptor *renderPass = view.currentRenderPassDescriptor;
  id<CAMetalDrawable> drawable = view.currentDrawable;
  if (!renderPass || !drawable)
    return;
  id<MTLBuffer> vertexBuffer;
  NSUInteger particleCount = 0;
  ViewerUniforms uniforms{};
  @synchronized(self) {
    _viewSize = view.drawableSize;
    vertexBuffer = _vertexBuffer;
    particleCount = _particleCount;
    if (particleCount > 0)
      uniforms = [self uniforms];
  }
  id<MTLCommandBuffer> commands = [_commandQueue commandBuffer];
  id<MTLRenderCommandEncoder> encoder =
      [commands renderCommandEncoderWithDescriptor:renderPass];
  [encoder setRenderPipelineState:_pipelineState];
  if (particleCount > 0) {
    [encoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
    [encoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypePoint
                vertexStart:0
                vertexCount:particleCount];
  }
  [encoder endEncoding];
  [commands presentDrawable:drawable];
  [commands commit];
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
  (void)view;
  @synchronized(self) {
    _viewSize = size;
  }
}
@end

@interface NBodyMTKView : MTKView
@property(nonatomic, weak) MetalRenderer *particleRenderer;
@end

@implementation NBodyMTKView {
  NSPoint _lastDragPoint;
  NSPoint _mouseDownPoint;
  CGFloat _dragDistanceSquared;
}
- (simd_float2)zoomAnchorForEvent:(NSEvent *)event {
  const NSPoint point =
      [self convertPoint:event.locationInWindow fromView:nil];
  const NSRect bounds = self.bounds;
  const float width = std::max(1.0, static_cast<double>(NSWidth(bounds)));
  const float height = std::max(1.0, static_cast<double>(NSHeight(bounds)));
  const float x = std::clamp(
      static_cast<float>(2.0 * (point.x - NSMinX(bounds)) / width - 1.0),
      -1.0F, 1.0F);
  const float y = std::clamp(
      static_cast<float>(2.0 * (point.y - NSMinY(bounds)) / height - 1.0),
      -1.0F, 1.0F);
  return simd_make_float2(x, y);
}
- (float)viewAspect {
  const NSSize size = self.bounds.size;
  return static_cast<float>(size.width / std::max(1.0, size.height));
}
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)acceptsFirstMouse:(NSEvent *)event {
  (void)event;
  return YES;
}
- (void)keyDown:(NSEvent *)event {
  [self.particleRenderer handleKeyDown:event];
}
- (void)scrollWheel:(NSEvent *)event {
  [self.particleRenderer handleScroll:event
                            viewHeight:NSHeight(self.bounds)];
}
- (void)magnifyWithEvent:(NSEvent *)event {
  [self.particleRenderer handleMagnify:event
                                 anchor:[self zoomAnchorForEvent:event]
                                 aspect:[self viewAspect]];
}
- (void)mouseDown:(NSEvent *)event {
  _mouseDownPoint = [self convertPoint:event.locationInWindow fromView:nil];
  _lastDragPoint = _mouseDownPoint;
  _dragDistanceSquared = 0.0;
}
- (void)mouseDragged:(NSEvent *)event {
  const NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
  const CGFloat deltaX = point.x - _lastDragPoint.x;
  const CGFloat deltaY = point.y - _lastDragPoint.y;
  _dragDistanceSquared += deltaX * deltaX + deltaY * deltaY;
  [self.particleRenderer handleDragX:deltaX deltaY:deltaY];
  _lastDragPoint = point;
}
- (void)mouseUp:(NSEvent *)event {
  const NSPoint point = [self convertPoint:event.locationInWindow fromView:nil];
  const CGFloat displacementX = point.x - _mouseDownPoint.x;
  const CGFloat displacementY = point.y - _mouseDownPoint.y;
  const CGFloat displacementSquared =
      displacementX * displacementX + displacementY * displacementY;
  if (_dragDistanceSquared <= 25.0 && displacementSquared <= 25.0) {
    [self.particleRenderer selectParticleAt:[self zoomAnchorForEvent:event]
                                  viewSize:self.bounds.size];
  }
}
@end

@interface FrameWatcher : NSObject
- (instancetype)initWithDirectory:(fs::path)directory
                          renderer:(MetalRenderer *)renderer
                            window:(NSWindow *)window
                  pollMilliseconds:(double)pollMilliseconds
             replayFramesPerSecond:(double)replayFramesPerSecond
                        replayOnly:(BOOL)replayOnly;
- (void)start;
@end

@implementation FrameWatcher {
  fs::path _directory;
  fs::path _displayedPath;
  __weak MetalRenderer *_renderer;
  __weak NSWindow *_window;
  double _pollSeconds;
  double _replaySeconds;
  NSTimer *_timer;
  std::string _lastError;
  std::string _generation;
  std::string _session;
  std::vector<fs::path> _replayFrames;
  std::size_t _replayIndex;
  std::chrono::steady_clock::time_point _nextReplay;
  bool _streamEnded;
  bool _replayOnly;
  fs::file_time_type _lastDirectoryWrite;
  bool _hasDirectoryWrite;
}
- (instancetype)initWithDirectory:(fs::path)directory
                          renderer:(MetalRenderer *)renderer
                            window:(NSWindow *)window
                  pollMilliseconds:(double)pollMilliseconds
             replayFramesPerSecond:(double)replayFramesPerSecond
                        replayOnly:(BOOL)replayOnly {
  self = [super init];
  if (self) {
    _directory = fs::absolute(std::move(directory)).lexically_normal();
    _renderer = renderer;
    _window = window;
    _pollSeconds = pollMilliseconds / 1000.0;
    _replaySeconds = 1.0 / replayFramesPerSecond;
    _replayIndex = 0;
    _nextReplay = std::chrono::steady_clock::now();
    _streamEnded = false;
    _replayOnly = replayOnly;
    _lastDirectoryWrite = fs::file_time_type{};
    _hasDirectoryWrite = false;
  }
  return self;
}
- (void)start {
  [self poll:nil];
  _timer = [NSTimer scheduledTimerWithTimeInterval:_pollSeconds
                                           target:self
                                         selector:@selector(poll:)
                                         userInfo:nil
                                          repeats:YES];
  _timer.tolerance = std::min(_pollSeconds * 0.25, 0.01);
}
- (void)poll:(NSTimer *)timer {
  (void)timer;
  @autoreleasepool {
    try {
      const std::optional<StreamState> state =
          _replayOnly ? std::nullopt : readStreamState(_directory);
      if (state && state->generation != _generation) {
        _generation = state->generation;
        _displayedPath.clear();
        _session.clear();
        _replayFrames.clear();
        _replayIndex = 0;
        _nextReplay = std::chrono::steady_clock::now();
        [_renderer clearSelection];
      }
      _streamEnded = _replayOnly || (state && state->ended);

      std::optional<fs::path> frame =
          _replayOnly ? newestSnapshot(_directory) : markerSnapshot(_directory);
      if (!_streamEnded && !frame && !state) {
        std::error_code timeError;
        const auto directoryWrite = fs::last_write_time(_directory, timeError);
        if (!timeError &&
            (!_hasDirectoryWrite || directoryWrite != _lastDirectoryWrite)) {
          _lastDirectoryWrite = directoryWrite;
          _hasDirectoryWrite = true;
          frame = newestSnapshot(_directory);
        }
      }

      if (!_streamEnded) {
        if (!frame || *frame == _displayedPath)
          return;
        MetalRenderer *renderer = _renderer;
        NSWindow *window = _window;
        if (!renderer || !window)
          return;
        const std::optional<std::string> session = frameSession(*frame);
        if (session && *session != _session) {
          if (!_session.empty())
            [renderer clearSelection];
          _session = *session;
          _replayFrames.clear();
          _replayIndex = 0;
        }
        Snapshot snapshot = loadStreamSnapshot(*frame);
        [renderer displaySnapshot:snapshot];
        window.title = snapshotTitle(snapshot);
        _displayedPath = *frame;
        _lastError.clear();
        return;
      }

      if (_session.empty() && frame) {
        const std::optional<std::string> session = frameSession(*frame);
        if (session)
          _session = *session;
      }
      if (_session.empty())
        return;
      if (_replayFrames.empty()) {
        _replayFrames = sessionSnapshots(_directory, _session);
        _replayIndex = 0;
        _nextReplay = std::chrono::steady_clock::now();
      }
      if (_replayFrames.empty())
        return;

      const auto now = std::chrono::steady_clock::now();
      if (now < _nextReplay)
        return;
      const fs::path replayPath = _replayFrames[_replayIndex];
      _replayIndex = (_replayIndex + 1) % _replayFrames.size();
      _nextReplay = now + std::chrono::duration_cast<
                              std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(_replaySeconds));
      if (replayPath == _displayedPath && _replayFrames.size() == 1)
        return;

      Snapshot snapshot = loadStreamSnapshot(replayPath);
      MetalRenderer *renderer = _renderer;
      NSWindow *window = _window;
      if (!renderer || !window)
        return;
      [renderer displaySnapshot:snapshot];
      window.title = snapshotTitle(snapshot, true);
      _displayedPath = replayPath;
      _lastError.clear();
    } catch (const std::exception &error) {
      if (_lastError != error.what()) {
        std::cerr << "Viewer is waiting for a valid frame: " << error.what()
                  << '\n';
        _lastError = error.what();
      }
    }
  }
}
@end

@interface NBodyAppDelegate : NSObject <NSApplicationDelegate>
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) MetalRenderer *renderer;
@property(nonatomic, strong) FrameWatcher *watcher;
@end

@implementation NBodyAppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
  (void)sender;
  return YES;
}
@end

int runViewer(const ViewerOptions &options) {
  @autoreleasepool {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      throw std::runtime_error("This Mac does not provide a Metal device");
    NSError *error = nil;
    MetalRenderer *renderer =
        [[MetalRenderer alloc] initWithDevice:device error:&error];
    if (!renderer) {
      const char *description = error.localizedDescription.UTF8String;
      throw std::runtime_error("Could not initialize Metal: " +
                               std::string(description ? description
                                                       : "unknown error"));
    }

    const NSRect frame = NSMakeRect(0, 0, 1000, 720);
    NSWindow *window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:NSWindowStyleMaskTitled |
                            NSWindowStyleMaskClosable |
                            NSWindowStyleMaskMiniaturizable |
                            NSWindowStyleMaskResizable
                    backing:NSBackingStoreBuffered
                      defer:NO];
    window.title = @"N-Body Live — waiting for frames";
    window.minSize = NSMakeSize(480, 320);
    [window center];

    NBodyMTKView *view = [[NBodyMTKView alloc] initWithFrame:frame
                                                     device:device];
    view.clearColor = MTLClearColorMake(0.002, 0.004, 0.012, 1.0);
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.preferredFramesPerSecond = 60;
    view.delegate = renderer;
    view.particleRenderer = renderer;
    window.contentView = view;
    [window makeFirstResponder:view];

    __attribute__((objc_precise_lifetime)) NBodyAppDelegate *delegate =
        [[NBodyAppDelegate alloc] init];
    delegate.window = window;
    delegate.renderer = renderer;
    NSApp.delegate = delegate;
    if (options.file) {
      Snapshot snapshot = loadSnapshot(*options.file);
      [renderer displaySnapshot:snapshot];
      window.title = snapshotTitle(snapshot);
    } else {
      const bool replayOnly = options.replayDirectory.has_value();
      const fs::path directory =
          replayOnly ? *options.replayDirectory : *options.streamDirectory;
      if (replayOnly) {
        std::error_code directoryError;
        if (!fs::is_directory(directory, directoryError) || directoryError)
          throw std::invalid_argument(
              "Replay directory does not exist: " + directory.string());
        window.title = @"N-Body Replay — loading frames";
      } else {
        fs::create_directories(directory);
      }
      delegate.watcher = [[FrameWatcher alloc]
          initWithDirectory:directory
                    renderer:renderer
                      window:window
            pollMilliseconds:options.pollMilliseconds
       replayFramesPerSecond:options.replayFramesPerSecond
                  replayOnly:replayOnly];
      [delegate.watcher start];
    }

    [window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    [NSApp run];
  }
  return 0;
}

int validate(const ViewerOptions &options) {
  fs::path path;
  if (options.file) {
    path = *options.file;
  } else {
    const fs::path &directory = options.replayDirectory
                                    ? *options.replayDirectory
                                    : *options.streamDirectory;
    std::optional<fs::path> latest = options.replayDirectory
                                         ? std::nullopt
                                         : markerSnapshot(directory);
    if (!latest)
      latest = newestSnapshot(directory);
    if (!latest)
      throw std::runtime_error("No complete .nbsnap frame is available in " +
                               directory.string());
    path = *latest;
  }
  const Snapshot snapshot = loadSnapshot(path);
  std::cout << "Loaded " << snapshot.header.particleCount << " particles";
  if (snapshot.streamed)
    std::cout << ", step " << snapshot.header.simulationStep << ", time "
              << snapshot.header.simulationTime;
  std::cout << " from " << path << '\n';
  return 0;
}

int main(int argc, char **argv) {
  try {
    const ViewerOptions options = parseOptions(argc, argv);
    return options.validateOnly ? validate(options) : runViewer(options);
  } catch (const std::invalid_argument &error) {
    std::cerr << "Argument error: " << error.what() << "\n\n";
    printUsage(std::cerr, argv[0]);
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "Viewer failed: " << error.what() << '\n';
    return 1;
  }
}
