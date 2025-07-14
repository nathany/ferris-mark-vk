#version 460

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texSampler;

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);

    // Multiply texture color by sprite color for tinting/alpha
    outColor = texColor * fragColor;

    // Optional: early fragment discard for fully transparent pixels
    // Note: This can hurt performance on tile-based GPUs, but helps with overdraw
    if (outColor.a < 0.01) {
        discard;
    }
}
