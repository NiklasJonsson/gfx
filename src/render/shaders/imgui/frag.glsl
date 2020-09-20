#version 450 core
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(location = 0) in vec4 f_color;
layout(location = 1) in vec2 f_uv;

void main()
{
    color = f_color * texture(u_texture, f_uv);
}