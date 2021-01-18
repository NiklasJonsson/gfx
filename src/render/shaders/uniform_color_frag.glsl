#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

#define MAX_NUM_PUNCTUAL_LIGHTS (16)
#define PUNCTUAL_LIGHTS_BITS (0xF)
struct PunctualLight {
    vec4 pos_dir;
    vec4 color_range; // .w is the range
};

layout(set = 0, binding = 1) uniform LightingData {
    PunctualLight punctual_lights[MAX_NUM_PUNCTUAL_LIGHTS];
    uint num_lights;
} lighting_data;
layout(set = 1, binding = 0) uniform UnlitUniformData {
    vec4 color;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = ubo.color;
}
