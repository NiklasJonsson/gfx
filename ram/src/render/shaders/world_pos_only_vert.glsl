#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = view_data.view_proj * vec4(position, 1.0);
}
