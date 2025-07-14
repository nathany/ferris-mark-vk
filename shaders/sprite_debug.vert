#version 460

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    uint spriteIndex;
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

    // For debugging, create a simple transform
    // Place sprite at instance position with fixed size
    float spriteWidth = 99.0;
    float spriteHeight = 70.0;
    float x = float(gl_InstanceIndex) * 120.0; // Spread sprites horizontally
    float y = 100.0; // Fixed Y position

    vec2 worldPos = vec2(x, y) + baseCoord * vec2(spriteWidth, spriteHeight);

    // Transform to clip space
    gl_Position = pc.viewProj * vec4(worldPos, 0.0, 1.0);

    // Simple UV mapping
    fragTexCoord = baseCoord;

    // White color
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
