layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

// 15 spotlights and 1 directional
#define MAX_NUM_SHADOWS (16)

vec4 world_to_clip(vec3 world_pos) {
    return view_data.view_proj * vec4(world_pos, 1.0);
}

vec4 world_to_clip(vec4 world_pos) {
    return view_data.view_proj * world_pos;
}


