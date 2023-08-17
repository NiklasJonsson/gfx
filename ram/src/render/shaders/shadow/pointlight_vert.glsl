#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShadowLightInfo {
    mat4 view_proj;
    vec4 pos;
} light_info;

layout(push_constant) uniform Model {
    mat4 model;
    mat4 model_it;
} model_tfm;
 
layout(location = 0) in vec3 position;

layout(location = 0) out VsOut {
    vec3 world_pos;
} vs_out;

void main()
{
    vs_out.world_pos = (model_tfm.model * vec4(position, 1.0)).xyz;
    gl_Position = light_info.view_proj * vec4(vs_out.world_pos, 1.0);
}