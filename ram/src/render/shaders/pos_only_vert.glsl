#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform PosOnlyViewData {
    mat4 view_proj;
} view_data;

layout(push_constant) uniform Model {
    mat4 model;
    mat4 model_it;
} model_tfm;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = view_data.view_proj * model_tfm.model * vec4(position, 1.0);
}
