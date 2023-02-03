#version 450
#extension GL_ARB_separate_shader_objects : enable

#define MAX_NUM_LIGHTS (16)

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

layout(push_constant) uniform Model {
    mat4 model;
    mat4 model_it; // inverse transpose of model matrix
} model_tfm;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out VsOut {
    vec3 world_normal;
    vec3 world_pos;
} vs_out;

void main() {
    vs_out.world_normal = normalize((model_tfm.model_it * vec4(normal, 0.0)).xyz);
    vs_out.world_pos = (model_tfm.model * vec4(position, 1.0)).xyz;
    gl_Position = view_data.view_proj * vec4(vs_out.world_pos, 1.0);
}
