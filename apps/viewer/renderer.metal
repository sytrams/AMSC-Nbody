#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    uint type [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float point_size [[point_size]];
    half4 color;
};

struct ViewerUniforms {
    float4x4 view_projection;
    float4 display;
};

enum ParticleType : uint {
    ParticleTypeUnknown = 0,
    ParticleTypeStar = 1,
    ParticleTypePlanet = 2,
    ParticleTypeMoon = 3,
    ParticleTypeAsteroid = 4,
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant ViewerUniforms &uniforms [[buffer(1)]],
                             uint vertex_id [[vertex_id]]) {
    VertexOut out;
    out.position = uniforms.view_projection * float4(in.position, 1.0);
    half depth = half(clamp(out.position.z, 0.0, 1.0));
    switch (in.type) {
        case ParticleTypeStar:
            out.point_size = max(uniforms.display.x * 10.0, 28.0);
            out.color = half4(1.0h, 0.78h, 0.16h, 1.0h);
            break;
        case ParticleTypePlanet:
            out.point_size = max(uniforms.display.x * 5.0, 12.0);
            out.color = half4(0.18h, 0.60h, 1.0h, 1.0h);
            break;
        case ParticleTypeMoon:
            out.point_size = max(uniforms.display.x * 3.0, 7.0);
            out.color = half4(0.78h, 0.84h, 0.92h, 0.98h);
            break;
        case ParticleTypeAsteroid:
            out.point_size = max(uniforms.display.x * 0.75, 1.0);
            out.color = mix(half4(0.58h, 0.32h, 0.16h, 0.52h),
                            half4(0.86h, 0.55h, 0.28h, 0.68h), depth);
            break;
        default:
            out.point_size = max(uniforms.display.x, 2.0);
            out.color = half4(0.86h, 0.92h, 1.0h, 0.85h);
            break;
    }
    if (uniforms.display.y >= 0.0 &&
        vertex_id == uint(uniforms.display.y)) {
        out.point_size = max(min(out.point_size * 1.8, 48.0), 14.0);
        out.color = half4(1.0h, 0.20h, 0.68h, 1.0h);
    }
    return out;
}

fragment half4 fragment_main(VertexOut in [[stage_in]],
                             float2 point_coord [[point_coord]]) {
    float distance_from_center = length(point_coord - float2(0.5));
    if (distance_from_center > 0.5) {
        discard_fragment();
    }
    half glow = half(1.0 - smoothstep(0.05, 0.5, distance_from_center));
    return half4(in.color.rgb * (0.72h + glow * 0.55h), in.color.a);
}
