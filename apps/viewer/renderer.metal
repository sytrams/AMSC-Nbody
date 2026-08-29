#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    uint type [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float point_size [[point_size]];
    float4 color;
};

enum ParticleType : uint {
    ParticleTypeStar = 1,
    ParticleTypePlanet = 2,
    ParticleTypeMoon = 3,
    ParticleTypeAsteroid = 4,
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant float4x4 &modelViewProjectionMatrix [[buffer(1)]]) {
    VertexOut out;
    out.position = modelViewProjectionMatrix * float4(in.position, 1.0);
    switch (in.type) {
        case ParticleTypeStar:
            out.point_size = 9.0;
            out.color = float4(1.0, 0.82, 0.25, 1.0);
            break;
        case ParticleTypePlanet:
            out.point_size = 7.0;
            out.color = float4(0.25, 0.60, 1.0, 1.0);
            break;
        case ParticleTypeMoon:
            out.point_size = 5.0;
            out.color = float4(0.78, 0.82, 0.88, 1.0);
            break;
        case ParticleTypeAsteroid:
            out.point_size = 2.0;
            out.color = float4(0.72, 0.48, 0.28, 1.0);
            break;
        default:
            out.point_size = 4.0;
            out.color = float4(1.0, 1.0, 1.0, 1.0);
            break;
    }
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return in.color;
}
