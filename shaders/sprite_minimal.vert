#version 460

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    uint spriteIndex;
} pc;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec4 fragColor;

void main() {
    // Test with hardcoded triangle vertices first
    vec2 positions[3] = vec2[](
        vec2(-0.5, -0.5),
        vec2( 0.5, -0.5),
        vec2( 0.0,  0.5)
    );

    vec2 texCoords[3] = vec2[](
        vec2(0.0, 1.0),
        vec2(1.0, 1.0),
        vec2(0.5, 0.0)
    );

    // Use vertex index to get position
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragTexCoord = texCoords[gl_VertexIndex];
    fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color to make it visible
}
