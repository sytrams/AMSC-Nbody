#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    float point_size [[point_size]];
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant float4x4 &modelViewProjectionMatrix [[buffer(1)]]) {
    VertexOut out;
    out.position = modelViewProjectionMatrix * float4(in.position, 1.0);
    out.point_size = 4.0;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(1.0, 1.0, 1.0, 1.0); // White points
}
