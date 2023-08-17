#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShadowLightInfo {
    mat4 view_proj;
    vec4 pos;
} light_info;

layout(location = 0) in VsOut {
    vec3 world_pos;
} vs_out;

layout(location = 0) out float out_color;

void main()
{
    out_color = length(vs_out.world_pos - light_info.pos.xyz);
}