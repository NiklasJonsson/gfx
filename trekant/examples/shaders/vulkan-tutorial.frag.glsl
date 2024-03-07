#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 1) uniform sampler2D u_color_map;

layout(location = 1) in vec2 frag_tex_coord;

layout(location = 0) out vec4 color;

void main() {
    color = texture(u_color_map, frag_tex_coord);
}
