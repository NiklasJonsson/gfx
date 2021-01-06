#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

layout(push_constant) uniform Model {
    mat4 model;
    mat4 model_it; // inverse transpose of model matrix
} model_tfm;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
#if HAS_TEX_COORDS
layout(location = TEX_COORDS_LOC) in vec2 tex_coords;
#endif

#if HAS_VERTEX_COLOR
layout(location = VCOL_LOC) in vec4 color;
#endif

#if HAS_TANGENTS
layout(location = TAN_LOC) in vec4 tangent;
#endif

layout(location = 0) out vec3 world_normal;
layout(location = 1) out vec3 world_pos;

#if HAS_TEX_COORDS
layout(location = TEX_COORDS_LOC) out vec2 tex_coords_0;
#endif

#if HAS_VERTEX_COLOR
layout(location = VCOL_LOC) out vec3 color_0;
#endif

#if HAS_TANGENTS
layout(location = TAN_LOC) out vec3 world_tangent;
layout(location = BITAN_LOC) out vec3 world_bitangent;
#endif

void main() {
    world_normal = normalize((model_tfm.model_it * vec4(normal, 0.0)).xyz);
    world_pos = (model_tfm.model * vec4(position, 1.0)).xyz;
#if HAS_TEX_COORDS
    tex_coords_0 = tex_coords;
#endif

#if HAS_VERTEX_COLOR
    color_0 = color.rgb;
#endif

#if HAS_TANGENTS
    world_tangent = normalize((model_tfm.model * vec4(tangent.xyz, 0.0)).xyz);
    world_bitangent = normalize(cross(world_normal, world_tangent) * tangent.w);
#endif

    gl_Position = view_data.view_proj * model_tfm.model * vec4(position, 1.0);
}
