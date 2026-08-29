#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <system_error>

#include "particle_type.hpp"

namespace {
constexpr const char* kDefaultViewerFile = "src/milky_way_particles_complete.bin";

struct ViewerFileHeader {
    uint64_t num_particles;
    std::streamoff header_bytes;
    bool has_particle_types;
};

ViewerFileHeader detect_viewer_file_header(std::ifstream& inFile) {
    inFile.clear();
    inFile.seekg(0, std::ios::end);
    const auto fileSize = inFile.tellg();
    if (fileSize < 0) {
        throw std::runtime_error("Failed to determine viewer file size.");
    }

    constexpr std::streamoff bytes_per_particle = static_cast<std::streamoff>(7 * sizeof(double));

    const auto matchesLayout = [&](uint64_t count, std::streamoff headerBytes, bool hasParticleTypes) {
        const auto typeBytes = hasParticleTypes ? static_cast<std::streamoff>(count) : std::streamoff{0};
        return fileSize == headerBytes + static_cast<std::streamoff>(count) * bytes_per_particle + typeBytes;
    };

    inFile.seekg(0, std::ios::beg);
    uint64_t n64 = 0;
    inFile.read(reinterpret_cast<char*>(&n64), sizeof(n64));
    if (inFile && n64 > 0) {
        const auto headerBytes = static_cast<std::streamoff>(sizeof(uint64_t));
        if (matchesLayout(n64, headerBytes, true)) {
            return {n64, headerBytes, true};
        }
        if (matchesLayout(n64, headerBytes, false)) {
            return {n64, headerBytes, false};
        }
    }

    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    uint32_t n32 = 0;
    inFile.read(reinterpret_cast<char*>(&n32), sizeof(n32));
    if (inFile && n32 > 0) {
        const auto headerBytes = static_cast<std::streamoff>(sizeof(uint32_t));
        if (matchesLayout(n32, headerBytes, true)) {
            return {n32, headerBytes, true};
        }
        if (matchesLayout(n32, headerBytes, false)) {
            return {n32, headerBytes, false};
        }
    }

    throw std::runtime_error("Unsupported viewer file layout. Expected planar doubles with a 32-bit or 64-bit particle count header and an optional uint8 particle-type block.");
}

struct ViewerOptions {
    std::string file = kDefaultViewerFile;
    std::uint64_t step = 0;
    std::optional<std::uint64_t> totalSteps;
};

bool parseUnsigned(std::string_view text, std::uint64_t& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool extract_viewer_args(int& argc, char** argv, ViewerOptions& options) {
    int write_index = 1;

    for (int read_index = 1; read_index < argc; ++read_index) {
        std::string arg = argv[read_index];

        if (arg == "-f" || arg == "--file") {
            if (read_index + 1 >= argc) {
                std::cerr << "Missing value for " << arg << '\n';
                return false;
            }

            options.file = argv[++read_index];
            continue;
        }

        constexpr std::string_view file_prefix = "--file=";
        if (arg.rfind(file_prefix.data(), 0) == 0) {
            options.file = arg.substr(file_prefix.size());
            continue;
        }

        if (arg == "--step" || arg == "--total-steps") {
            if (read_index + 1 >= argc) {
                std::cerr << "Missing value for " << arg << '\n';
                return false;
            }
            std::uint64_t value = 0;
            if (!parseUnsigned(argv[++read_index], value) ||
                (arg == "--total-steps" && value == 0)) {
                std::cerr << "Invalid value for " << arg << '\n';
                return false;
            }
            if (arg == "--step") {
                options.step = value;
            } else {
                options.totalSteps = value;
            }
            continue;
        }

        constexpr std::string_view step_prefix = "--step=";
        constexpr std::string_view total_steps_prefix = "--total-steps=";
        const bool is_step = arg.rfind(step_prefix.data(), 0) == 0;
        const bool is_total_steps = arg.rfind(total_steps_prefix.data(), 0) == 0;
        if (is_step || is_total_steps) {
            const auto prefix_size = is_step ? step_prefix.size() : total_steps_prefix.size();
            std::uint64_t value = 0;
            if (!parseUnsigned(std::string_view(arg).substr(prefix_size), value) ||
                (is_total_steps && value == 0)) {
                std::cerr << "Invalid value for "
                          << (is_step ? "--step" : "--total-steps") << '\n';
                return false;
            }
            if (is_step) {
                options.step = value;
            } else {
                options.totalSteps = value;
            }
            continue;
        }

        argv[write_index++] = argv[read_index];
    }

    argc = write_index;
    if (options.totalSteps && options.step > *options.totalSteps) {
        std::cerr << "--step cannot be greater than --total-steps\n";
        return false;
    }
    return true;
}
}  // namespace

struct ParticleData {
    simd_float3 position;
    std::uint32_t type;
};

// Internal loading class to handle different binary formats
class MilkyWayLoader {
public:
    uint64_t num_particles;
    std::unique_ptr<double[]> x, y, z;
    std::unique_ptr<std::uint8_t[]> type;

    explicit MilkyWayLoader(const char* file_path) {
        std::ifstream inFile(file_path, std::ios::binary);
        if (!inFile) throw std::runtime_error("Could not open file");

        const auto header = detect_viewer_file_header(inFile);
        num_particles = header.num_particles;
        x = std::make_unique<double[]>(num_particles);
        y = std::make_unique<double[]>(num_particles);
        z = std::make_unique<double[]>(num_particles);
        type = std::make_unique<std::uint8_t[]>(num_particles);

        inFile.clear();
        inFile.seekg(header.header_bytes, std::ios::beg);
        const auto bytes = static_cast<std::streamoff>(num_particles * sizeof(double));
        inFile.seekg(bytes, std::ios::cur); // mass
        inFile.read(reinterpret_cast<char*>(x.get()), bytes);
        inFile.read(reinterpret_cast<char*>(y.get()), bytes);
        inFile.read(reinterpret_cast<char*>(z.get()), bytes);

        if (!inFile) {
            throw std::runtime_error("Failed to read particle coordinates from viewer file.");
        }

        if (header.has_particle_types) {
            inFile.seekg(3 * bytes, std::ios::cur); // vx, vy, vz
            inFile.read(reinterpret_cast<char*>(type.get()), static_cast<std::streamsize>(num_particles));
            if (!inFile) {
                throw std::runtime_error("Failed to read particle types from viewer file.");
            }
            for (uint64_t index = 0; index < num_particles; ++index) {
                if (!isValidParticleType(type[index])) {
                    throw std::runtime_error("Viewer file contains an unsupported particle type.");
                }
            }
        } else {
            std::fill_n(type.get(), num_particles, static_cast<std::uint8_t>(ParticleType::Unknown));
        }
        std::cout << "Successfully loaded " << num_particles << " particles from " << file_path << std::endl;
    }
};

@interface MetalRenderer : NSObject <MTKViewDelegate>
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLRenderPipelineState> pipelineState;
@property (nonatomic, strong) id<MTLBuffer> vertexBuffer;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, assign) NSUInteger numParticles;

- (void)handleKeyDown:(NSEvent *)event;
- (void)handleScrollWheel:(NSEvent *)event;
@end

@implementation MetalRenderer {
    simd_float4x4 _projectionMatrix;
    float _maxCoord;
    simd_float2 _panOffset;
    float _zoomFactor;
    CGSize _viewSize;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device particles:(std::vector<ParticleData>&)particles {
    self = [super init];
    if (self) {
        _device = device;
        _numParticles = particles.size();
        _commandQueue = [device newCommandQueue];
        _panOffset = (simd_float2){0, 0};
        _zoomFactor = 1.0f;
        
        float maxRange = 0.001f;
        for (const auto& p : particles) {
            maxRange = std::max({maxRange, std::abs(p.position.x), std::abs(p.position.y), std::abs(p.position.z)});
        }
        _maxCoord = maxRange * 1.1f; // 10% padding
        std::cout << "Loaded " << _numParticles << " particles. Max coordinate: " << maxRange << std::endl;

        _vertexBuffer = [device newBufferWithBytes:particles.data()
                                            length:particles.size() * sizeof(ParticleData)
                                           options:MTLResourceStorageModeShared];
        
        [self setupPipeline];
    }
    return self;
}

- (void)setupPipeline {
    NSError *error = nil;
    NSString *shaderPath = @"apps/viewer/renderer.metal";
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
    if (!shaderSource) {
        NSLog(@"Failed to load shader source: %@", error);
        return;
    }
    
    id<MTLLibrary> library = [_device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        NSLog(@"Failed to create library: %@", error);
        return;
    }
    
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragment_main"];
    
    MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    
    MTLVertexDescriptor *vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[1].format = MTLVertexFormatUInt;
    vertexDescriptor.attributes[1].offset = offsetof(ParticleData, type);
    vertexDescriptor.attributes[1].bufferIndex = 0;
    vertexDescriptor.layouts[0].stride = sizeof(ParticleData);
    pipelineDescriptor.vertexDescriptor = vertexDescriptor;
    
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
}

- (void)updateProjection {
    float aspect = (_viewSize.height > 0) ? (_viewSize.width / _viewSize.height) : 1.0f;
    float s = _maxCoord / _zoomFactor;
    
    // Applying pan in world coordinates
    float tx = -_panOffset.x / (_maxCoord * aspect);
    float ty = -_panOffset.y / _maxCoord;

    _projectionMatrix = simd_matrix_from_rows(
        (simd_float4){1.0f/(s * aspect), 0, 0, tx * _zoomFactor},
        (simd_float4){0, 1.0f/s, 0, ty * _zoomFactor},
        (simd_float4){0, 0, 1.0f/s, 0},
        (simd_float4){0, 0, 0, 1.0f}
    );
}

- (void)handleKeyDown:(NSEvent *)event {
    float moveStep = _maxCoord * 0.05f / _zoomFactor;
    BOOL ctrlPressed = (event.modifierFlags & NSEventModifierFlagControl) != 0;
    NSString *chars = [event charactersIgnoringModifiers];
    
    if (event.keyCode == 123) { // Left
        _panOffset.x -= moveStep;
    } else if (event.keyCode == 124) { // Right
        _panOffset.x += moveStep;
    } else if (event.keyCode == 125) { // Down
        _panOffset.y -= moveStep;
    } else if (event.keyCode == 126) { // Up
        _panOffset.y += moveStep;
    } else if (ctrlPressed && ([chars isEqualToString:@"+"] || [chars isEqualToString:@"="])) {
        _zoomFactor *= 1.2f;
    } else if (ctrlPressed && [chars isEqualToString:@"-"]) {
        _zoomFactor /= 1.2f;
    }
}

- (void)handleScrollWheel:(NSEvent *)event {
    const float gestureScale = event.hasPreciseScrollingDeltas ? 1.0f : 12.0f;
    const float worldUnitsPerPoint =
        _maxCoord * 0.0025f * gestureScale / std::max(_zoomFactor, 0.001f);

    // Intentionally invert the vertical two-finger gesture direction.
    _panOffset.y -= static_cast<float>(event.scrollingDeltaY) * worldUnitsPerPoint;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    _viewSize = view.drawableSize;
    [self updateProjection];
    
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (!renderPassDescriptor) return;
    
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    [renderEncoder setRenderPipelineState:_pipelineState];
    [renderEncoder setVertexBuffer:_vertexBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBytes:&_projectionMatrix length:sizeof(_projectionMatrix) atIndex:1];
    [renderEncoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:_numParticles];
    [renderEncoder endEncoding];
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    _viewSize = size;
}
@end

@interface NBodyMTKView : MTKView
@property (nonatomic, assign) MetalRenderer *renderer;
@end

@implementation NBodyMTKView
- (BOOL)acceptsFirstResponder { return YES; }
- (void)keyDown:(NSEvent *)event {
    [_renderer handleKeyDown:event];
}
- (void)scrollWheel:(NSEvent *)event {
    [_renderer handleScrollWheel:event];
}
@end

void run_metal_viewer(const ViewerOptions& options) {
    try {
        MilkyWayLoader loader(options.file.c_str());
        std::vector<ParticleData> particles(loader.num_particles);
        for (size_t i = 0; i < loader.num_particles; ++i) {
            particles[i].position = (simd_float3){ (float)loader.x[i], (float)loader.y[i], (float)loader.z[i] };
            particles[i].type = loader.type[i];
        }
        
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        
        NSRect frame = NSMakeRect(0, 0, 800, 600);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        NSString *title = nil;
        if (options.totalSteps) {
            title = [NSString stringWithFormat:
                @"N-Body Viewer (Interactive Mode) - Step %llu/%llu",
                static_cast<unsigned long long>(options.step),
                static_cast<unsigned long long>(*options.totalSteps)];
        } else {
            title = [NSString stringWithFormat:
                @"N-Body Viewer (Interactive Mode) - Step %llu/?",
                static_cast<unsigned long long>(options.step)];
        }
        [window setTitle:title];
        [window makeKeyAndOrderFront:nil];
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NBodyMTKView *view = [[NBodyMTKView alloc] initWithFrame:frame device:device];
        view.clearColor = MTLClearColorMake(0, 0, 0, 1);
        view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        
        MetalRenderer *renderer = [[MetalRenderer alloc] initWithDevice:device particles:particles];
        view.delegate = renderer;
        view.renderer = renderer;
        
        [window setContentView:view];
        [window makeFirstResponder:view];
        [NSApp activateIgnoringOtherApps:YES];
        [NSApp run];
    } catch (const std::exception& e) {
        std::cerr << "Error in viewer: " << e.what() << std::endl;
    }
}

int main(int argc, char **argv) {
    ViewerOptions options;
    if (!extract_viewer_args(argc, argv, options)) {
        return 2;
    }

    run_metal_viewer(options);
    return 0;
}
