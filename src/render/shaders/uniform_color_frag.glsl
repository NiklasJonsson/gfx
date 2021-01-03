#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 1) uniform LightingData {
    vec4 light_pos;
    vec4 view_pos;
} lighting_data;

layout(set = 1, binding = 0) uniform Color {
    vec4 x;
} color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = color.x;
}
