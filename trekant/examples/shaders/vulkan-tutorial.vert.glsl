#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
	mat4 model;
	mat4 view;
	mat4 proj;
} u_matrices;

layout(location = 0) in vec3 pos;
layout(location = 2) in vec2 tex_coord;

layout(location = 1) out vec2 frag_tex_coord;

void main() {
	gl_Position = u_matrices.proj * u_matrices.view * u_matrices.model * vec4(pos, 1.0);
	frag_tex_coord = tex_coord;
}
