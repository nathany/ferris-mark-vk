#version 460

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D texSampler;

void main() {
    vec4 texColor = texture(texSampler, fragTexCoord);

    // Multiply texture color by sprite color for tinting/alpha
    outColor = texColor * fragColor;

    // Removed alpha discard for better performance on tile-based GPUs
    // Modern GPUs handle transparent pixels efficiently
}
