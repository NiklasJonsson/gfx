#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform Transforms {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 0, binding = 1) uniform Model {
    mat4 model;
    mat4 model_it;
} model_ubo;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = ubo.proj * ubo.view * model_ubo.model * vec4(position, 1.0);
}
