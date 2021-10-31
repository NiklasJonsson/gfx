#version 450
#extension GL_ARB_separate_shader_objects : enable

#define MAX_NUM_LIGHTS (16)

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

layout(set = 0, binding = 3) uniform ShadowMatrices {
    mat4 matrices[MAX_NUM_LIGHTS];
    uvec4 num_matrices;
} shadow_matrices;

uint num_shadow_matrices() {
    return min(MAX_NUM_LIGHTS, shadow_matrices.num_matrices.x);
}

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

layout(location = 0) out VsOut {
    vec3 world_normal;
    vec3 world_pos;
    vec4 shadow_coords[MAX_NUM_LIGHTS];

#if HAS_TEX_COORDS
    vec2 tex_coords_0;
#endif

#if HAS_VERTEX_COLOR
    vec3 color_0;
#endif

#if HAS_TANGENTS
    vec3 world_tangent;
    vec3 world_bitangent;
#endif
} vs_out;

// Map clip space coords [-w, w] to [0, w] so that perspective divide (done in fragment shader)
// transforms it into [0, 1] (unit interval) which we can use to sample the shadow map.
const mat4 clip_to_unit = mat4(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);

void main() {
    vs_out.world_normal = normalize((model_tfm.model_it * vec4(normal, 0.0)).xyz);
    vs_out.world_pos = (model_tfm.model * vec4(position, 1.0)).xyz;
#if HAS_TEX_COORDS
    vs_out.tex_coords_0 = tex_coords;
#endif

#if HAS_VERTEX_COLOR
    vs_out.color_0 = color.rgb;
#endif

#if HAS_TANGENTS
    vs_out.world_tangent = normalize((model_tfm.model * vec4(tangent.xyz, 0.0)).xyz);
    vs_out.world_bitangent = normalize(cross(vs_out.world_normal, vs_out.world_tangent) * tangent.w);
#endif

    for (int i = 0; i < num_shadow_matrices(); ++i) {
        vs_out.shadow_coords[i] = clip_to_unit * shadow_matrices.matrices[i] * vec4(vs_out.world_pos, 1.0);
    }

    gl_Position = view_data.view_proj * vec4(vs_out.world_pos, 1.0);
}
