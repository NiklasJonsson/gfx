#version 450 core
// From the imgui example implementation

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(push_constant) uniform uPushConstant { vec2 scale; vec2 translate; } pc;

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec2 f_uv;

void main()
{
    f_color = color;
    f_uv = uv;
    gl_Position = vec4(pos * pc.scale + pc.translate, 0, 1);
}