#define M_PI (3.1415926535897932384626433832795)

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

#define SHADOW_TYPE_INVALID (0)
#define SHADOW_TYPE_DIRECTIONAL (1)
#define SHADOW_TYPE_SPOT (2)
#define SHADOW_TYPE_POINT (3)

#define MAX_NUM_LIGHTS (16)
struct PackedLight {
    vec4 pos;
    vec4 dir_cutoff;
    vec4 color_range; // .w is the range
    uvec4 shadow_info; // x is the shadow type, y is the shadow index in that array
};

layout(std140, set = 0, binding = 1) readonly buffer WorldToShadow {
    mat4 data[];
} world_to_shadow;

layout(set = 0, binding = 2) uniform LightingData {
    PackedLight lights[MAX_NUM_LIGHTS];
    vec4 ambient; // vec3 color + float strength
    uvec4 num_lights; // The number of lights in the array
} lighting_data;

uint num_lights() {
    return min(lighting_data.num_lights.x, MAX_NUM_LIGHTS);
}

// TODO: Make this a 1-elem array to simplify?
layout(set = 0, binding = 3) uniform sampler2D directional_shadow_map;

#define NUM_SPOTLIGHT_SHADOW_MAPS (16)
layout(set = 0, binding = 4) uniform sampler2D spotlight_shadow_maps[NUM_SPOTLIGHT_SHADOW_MAPS];

#define NUM_POINTLIGHT_SHADOW_MAPS (16)
layout(set = 0, binding = 5) uniform samplerCube pointlight_shadow_maps[NUM_POINTLIGHT_SHADOW_MAPS];

struct ShadowInfo {
    uint type;
    uint coords_idx;
    uint texture_idx;
};

struct Light {
    vec3 color;
    float attenuation;
    // TODO: Rename to normalized_direction
    vec3 direction;
    vec3 position;
    ShadowInfo shadow_info;
};

// This is based on the recommended impl for KHR_punctual_lights:
// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
// It does not contain the square for the smooth factor though.
// This frostbite presentation has some more details:
// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
// Also in real-time rendering 4, p 113, eq. 5.14 (which refers to the above presentation)
float distance_attenuation(vec3 light_vec, float light_range) {
    float dist_sqr = dot(light_vec, light_vec);
    float range_sqr = pow(light_range, 2.0);
    // inverse square law for light falloff & eps to avoid singularity
    float attenuation = 1.0 / max(dist_sqr, pow(0.01, 2.0));
    // Modified lerp(attenuation, 0, dist / light_range) so that we get a maximum range but the lerp does not affect
    // the falloff too severely. See frostbite presentation/real-time rendering for more info.
    float smooth_factor = pow(clamp(1.0 - pow(dist_sqr / range_sqr, 2.0), 0.0, 1.0), 2.0);

    return attenuation * smooth_factor;
}

float remap(float x, float old_low, float old_high, float new_low, float new_high) {
    return ((x - old_low) / (old_high - old_low)) * (new_high - new_low) + new_low;
}

Light unpack_light(PackedLight l, vec3 world_pos) {
    Light r;
    r.color = l.color_range.xyz;
    r.shadow_info.type = l.shadow_info.x;
    r.shadow_info.coords_idx = l.shadow_info.y;
    r.shadow_info.texture_idx = l.shadow_info.z;
    if (l.pos.w == 0.0) {
        // Directional
        r.direction = normalize(-l.dir_cutoff.xyz);
        r.attenuation = 1.0;
        r.position = vec3(0.0);
    } else if (l.dir_cutoff.w == 0.0) {
        // Point
        vec3 direction_unnormalized = l.pos.xyz - world_pos;
        r.direction = normalize(direction_unnormalized);
        r.attenuation = distance_attenuation(direction_unnormalized, l.color_range.w);
        r.position = vec3(l.pos.xyz);
    } else {
        // Spot
        vec3 direction_unnormalized = l.pos.xyz - world_pos;
        r.direction = normalize(direction_unnormalized);
        r.position = vec3(l.pos.xyz);
        r.attenuation = 0.0;
        vec3 spot_dir = normalize(-l.dir_cutoff.xyz);
        float cos_angle = dot(r.direction, spot_dir);
        if (cos_angle > l.dir_cutoff.w) {
            r.attenuation = distance_attenuation(direction_unnormalized, l.color_range.w);
            // Cone attenuation
            float cos_angle_norm = remap(cos_angle, l.dir_cutoff.w, 1.0, 0.0, 1.0); 
            // This is a function that has a steep incline at first and then goes to 1.0
            // Making this depend on cutoff means that the wider the code, the earlier the cutoff starts
            r.attenuation *= smoothstep(l.dir_cutoff.w, 1.0, cos_angle_norm + l.dir_cutoff.w);
        }
    }
    return r;
}

bool light_has_shadow(Light l) {
    return l.shadow_info.type != SHADOW_TYPE_INVALID;
}

float sample_depth_shadow_texture(sampler2D tex, vec3 coords, float n_dot_l) {
    float sample_depth = texture(tex, coords.xy).r;

    if (coords.z > 1.0 || coords.z < -1.0) {
        return 1.0;
    }

    float max_bias = 0.05;
    float min_bias = 0.005;
    float bias = max(max_bias * (1.0 - n_dot_l), min_bias); 
    return (coords.z - bias) < sample_depth ? 1.0 : 0.0;
}

float sample_shadow_map(vec3 coords, ShadowInfo info, vec3 frag_to_light_ls, float n_dot_l) {
    if (info.type == SHADOW_TYPE_DIRECTIONAL) {
        return sample_depth_shadow_texture(directional_shadow_map, coords, n_dot_l);
    } else if (info.type == SHADOW_TYPE_SPOT) {
        return sample_depth_shadow_texture(spotlight_shadow_maps[info.texture_idx], coords, n_dot_l);
    } else {
        // TODO: Bias here as well
        vec3 light_to_frag_ls = -frag_to_light_ls;
        float sample_depth = texture(pointlight_shadow_maps[info.texture_idx], light_to_frag_ls).r;
        return length(light_to_frag_ls) > sample_depth ? 0.0 : 1.0;
    }
}

// Map clip space coords [-w, w] to [0, w] so that perspective divide
// transforms it into [0, 1] (unit interval) which we can use to sample the shadow map.
// (Embarassing note to self: Column-major order of arguments, meaning the first 4 args are the first column).
const mat4 CLIP_BIAS = mat4(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);

float compute_shadow_factor(vec3 fragment_world_pos, Light light, float n_dot_l) {
    float shadow_factor = 1.0;
    if (light_has_shadow(light)) {
        ShadowInfo info = light.shadow_info;
        vec4 fragment_shadow_pos = CLIP_BIAS * world_to_shadow.data[info.coords_idx] * vec4(fragment_world_pos, 1.0);
        vec3 coords = fragment_shadow_pos.xyz / fragment_shadow_pos.w;
        vec4 frag_to_light_ls = world_to_shadow.data[info.coords_idx] * vec4(light.position - fragment_world_pos, 0.0);
        shadow_factor = sample_shadow_map(coords, info, frag_to_light_ls.xyz, n_dot_l);
    }
    return shadow_factor;
}
