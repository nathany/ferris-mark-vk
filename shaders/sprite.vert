#version 460

struct SpriteCommand {
    mat4 transform;
    vec4 color;
    vec2 uvMin;
    vec2 uvMax;
};

layout(set = 0, binding = 1, std430) readonly buffer SpriteBuffer {
    SpriteCommand commands[];
} spriteBuffer;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec4 fragColor;

void main() {
    // Procedural quad generation using gl_VertexIndex
    // This generates vertices for two triangles forming a quad:
    // Triangle 1: (0,0), (1,0), (0,1)
    // Triangle 2: (0,1), (1,0), (1,1)
    vec2 vertices[6] = vec2[](
        vec2(0.0, 0.0), // bottom-left
        vec2(1.0, 0.0), // bottom-right
        vec2(0.0, 1.0), // top-left
        vec2(0.0, 1.0), // top-left
        vec2(1.0, 0.0), // bottom-right
        vec2(1.0, 1.0)  // top-right
    );
    vec2 baseCoord = vertices[gl_VertexIndex];

    // Get sprite command for this instance
    SpriteCommand cmd = spriteBuffer.commands[gl_InstanceIndex];

    // Transform vertex position
    gl_Position = pc.viewProj * cmd.transform * vec4(baseCoord, 0.0, 1.0);

    // Interpolate UV coordinates
    fragTexCoord = mix(cmd.uvMin, cmd.uvMax, baseCoord);

    // Pass through color for tinting/alpha
    fragColor = cmd.color;
}
