#version 450
#extension GL_ARB_separate_shader_objects : enable

#include <engine.vert.glsl>

layout(push_constant) uniform Model {
    mat4 model;
    mat4 model_it;
} model_tfm;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = world_to_clip(model_tfm.model * vec4(position, 1.0));
}
