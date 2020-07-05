#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform Color {
    vec4 x;
} color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = color.x;
}
