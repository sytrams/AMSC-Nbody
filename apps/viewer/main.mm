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

namespace {
constexpr const char* kDefaultViewerFile = "src/milky_way_particles_complete.bin";

struct ViewerFileHeader {
    uint64_t num_particles;
    std::streamoff header_bytes;
};

ViewerFileHeader detect_viewer_file_header(std::ifstream& inFile) {
    inFile.clear();
    inFile.seekg(0, std::ios::end);
    const auto fileSize = inFile.tellg();
    if (fileSize < 0) {
        throw std::runtime_error("Failed to determine viewer file size.");
    }

    constexpr std::streamoff bytes_per_particle = static_cast<std::streamoff>(7 * sizeof(double));

    inFile.seekg(0, std::ios::beg);
    uint64_t n64 = 0;
    inFile.read(reinterpret_cast<char*>(&n64), sizeof(n64));
    if (inFile
        && n64 > 0
        && fileSize == static_cast<std::streamoff>(sizeof(uint64_t)) + static_cast<std::streamoff>(n64) * bytes_per_particle) {
        return {n64, static_cast<std::streamoff>(sizeof(uint64_t))};
    }

    inFile.clear();
    inFile.seekg(0, std::ios::beg);
    uint32_t n32 = 0;
    inFile.read(reinterpret_cast<char*>(&n32), sizeof(n32));
    if (inFile
        && n32 > 0
        && fileSize == static_cast<std::streamoff>(sizeof(uint32_t)) + static_cast<std::streamoff>(n32) * bytes_per_particle) {
        return {n32, static_cast<std::streamoff>(sizeof(uint32_t))};
    }

    throw std::runtime_error("Unsupported viewer file layout. Expected planar doubles with a 32-bit or 64-bit particle count header.");
}

bool extract_viewer_file_arg(int& argc, char** argv, std::string& viewer_file) {
    int write_index = 1;

    for (int read_index = 1; read_index < argc; ++read_index) {
        std::string arg = argv[read_index];

        if (arg == "-f" || arg == "--file") {
            if (read_index + 1 >= argc) {
                std::cerr << "Missing value for " << arg << '\n';
                return false;
            }

            viewer_file = argv[++read_index];
            continue;
        }

        constexpr std::string_view file_prefix = "--file=";
        if (arg.rfind(file_prefix.data(), 0) == 0) {
            viewer_file = arg.substr(file_prefix.size());
            continue;
        }

        argv[write_index++] = argv[read_index];
    }

    argc = write_index;
    return true;
}
}  // namespace

struct ParticleData {
    simd_float3 position;
};

// Internal loading class to handle different binary formats
class MilkyWayLoader {
public:
    uint64_t num_particles;
    std::unique_ptr<double[]> x, y, z;

    explicit MilkyWayLoader(const char* file_path) {
        std::ifstream inFile(file_path, std::ios::binary);
        if (!inFile) throw std::runtime_error("Could not open file");

        const auto header = detect_viewer_file_header(inFile);
        num_particles = header.num_particles;
        x = std::make_unique<double[]>(num_particles);
        y = std::make_unique<double[]>(num_particles);
        z = std::make_unique<double[]>(num_particles);

        inFile.clear();
        inFile.seekg(header.header_bytes, std::ios::beg); inFile.seekg(num_particles * sizeof(double), std::ios::cur);

        size_t bytes = num_particles * sizeof(double);
        inFile.seekg(bytes, std::ios::cur);
        inFile.read(reinterpret_cast<char*>(x.get()), bytes);
        inFile.read(reinterpret_cast<char*>(y.get()), bytes);
        inFile.read(reinterpret_cast<char*>(z.get()), bytes);

        if (!inFile) {
            throw std::runtime_error("Failed to read particle coordinates from viewer file.");
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
@end

void run_metal_viewer(const char* file_path) {
    try {
        MilkyWayLoader loader(file_path);
        std::vector<ParticleData> particles(loader.num_particles);
        for (size_t i = 0; i < loader.num_particles; ++i) {
            particles[i].position = (simd_float3){ (float)loader.x[i], (float)loader.y[i], (float)loader.z[i] };
        }
        
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        
        NSRect frame = NSMakeRect(0, 0, 800, 600);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"N-Body Milky Way Viewer (Interactive Mode)"];
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
    std::string viewer_file = kDefaultViewerFile;
    if (!extract_viewer_file_arg(argc, argv, viewer_file)) {
        return 1;
    }

    run_metal_viewer(viewer_file.c_str());
    return 0;
}
