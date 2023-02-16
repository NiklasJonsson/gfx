#version 450
#extension GL_ARB_separate_shader_objects : enable

#include <engine/frag.glsl>

layout(set = 1, binding = 0) uniform UnlitUniformData {
    vec4 color;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = ubo.color;
}
