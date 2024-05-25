#version 450
#extension GL_ARB_separate_shader_objects : enable

// TODO: Share with common file.

layout(set = 0, binding = 0) uniform ShadowLightInfo {
    mat4 view_proj;
    vec4 pos;
} light_info;

layout(push_constant) uniform Model {
    mat4 model;
    mat4 model_it;
} model_tfm;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = light_info.view_proj * model_tfm.model * vec4(position, 1.0);
}
