layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

// 16 spotlights and 1 directional
#define MAX_NUM_SHADOWS (17)

layout(set = 0, binding = 3) uniform ShadowMatrices {
    mat4 matrices[];
    uvec4 num_matrices;
} shadow_matrices;

uint num_shadow_matrices() {
    return min(MAX_NUM_SHADOWS, shadow_matrices.num_matrices.x);
}

vec4 world_to_clip(vec3 world_pos) {
    return view_data.view_proj * vec4(world_pos, 1.0);
}

vec4 world_to_clip(vec4 world_pos) {
    return view_data.view_proj * world_pos;
}

// Map clip space coords [-w, w] to [0, w] so that perspective divide (done in fragment shader)
// transforms it into [0, 1] (unit interval) which we can use to sample the shadow map.
const mat4 clip_to_unit = mat4(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);

void write_shadow_coords(vec3 world_pos, out vec4 shadow_coords_out[MAX_NUM_SHADOWS]) {
    uint n = num_shadow_matrices();
    for (uint i = 0; i < n; ++i) {
        shadow_coords_out[i] = clip_to_unit * shadow_matrices.matrices[i] * vec4(world_pos, 1.0);
    }
}
