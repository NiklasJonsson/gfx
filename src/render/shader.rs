// TODO: Move pipeline creation here

pub mod vs_static {
    vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform VPUniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 matrix;
} model;

layout(location = 0) in vec3 position;
layout(location = 2) in vec2 tex_coords;

layout(location = 1) out vec2 frag_tex_coords;

void main() {
    gl_Position = ubo.proj * ubo.view * model.matrix * vec4(position, 1.0);
    frag_tex_coords = tex_coords;
}
"
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform VPUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;


layout(location = 0) in vec3 position;
layout(location = 2) in vec2 tex_coords;

layout(location = 1) out vec2 frag_tex_coords;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    frag_tex_coords = tex_coords;
}
"
}
}

pub mod fs {
    vulkano_shaders::shader! {
    ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 1) in vec2 frag_tex_coords;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, frag_tex_coords);
}
"
    }
}

pub mod vs_pbr_static {
    vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform VPUniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 matrix;
} model;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = ubo.proj * ubo.view * model.matrix * vec4(position, 1.0);
}
"
    }
}

// If there is no base color texture, the BaseColorFactor is the color
pub mod fs_pbr {
    vulkano_shaders::shader! {
    ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

// PBR uniforms
layout(binding = 1) uniform PBRMaterialData {
    vec4 BaseColorFactor;
} pbr_data;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = pbr_data.BaseColorFactor;
}
"
    }
}
