#version 450
#extension GL_ARB_separate_shader_objects : enable

#define M_PI (3.1415926535897932384626433832795)

layout(set = 0, binding = 0) uniform ViewData {
    mat4 view_proj;
    vec4 view_pos;
} view_data;

#define MAX_NUM_LIGHTS (16)
struct PackedLight {
    vec4 pos;
    vec4 dir_cutoff;
    vec4 color_range; // .w is the range
};

layout(set = 0, binding = 1) uniform LightingData {
    PackedLight lights[MAX_NUM_LIGHTS];
    vec4 ambient; // vec3 color + float strength
    uvec4 num_lights; // The number of lights in the array
} lighting_data;

uint num_lights() {
    return min(lighting_data.num_lights.x, MAX_NUM_LIGHTS);
}

struct Light {
    vec3 color;
    float attenuation;
    vec3 direction;
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
    if (l.pos.w == 0.0) {
        // Directional
        r.direction = normalize(-l.dir_cutoff.xyz);
        r.attenuation = 1.0;
    } else if (l.dir_cutoff.w == 0.0) {
        // Point
        vec3 direction_unnormalized = l.pos.xyz - world_pos;
        r.direction = normalize(direction_unnormalized);
        r.attenuation = distance_attenuation(direction_unnormalized, l.color_range.w);
    } else {
        // Spot
        vec3 direction_unnormalized = l.pos.xyz - world_pos;
        r.direction = normalize(direction_unnormalized);
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

layout(location = 0) in VsOut {
    vec3 world_normal;
    vec3 world_pos;
} vs_out;

layout(location = 0) out vec4 out_color;

layout(set = 1, binding = 0) uniform MaterialData {
    vec4 color;
} material_data;

void main() {
    vec3 normal = normalize(vs_out.world_normal);

    vec3 color = vec3(0);

    // Ambient
    color += lighting_data.ambient.xyz * material_data.color.xyz * lighting_data.ambient.w;

    // Lights
    uint nlights = num_lights();
    for (uint i = 0; i < nlights; ++i) {
        Light light = unpack_light(lighting_data.lights[i], vs_out.world_pos);
        float n_dot_l = clamp(dot(normal, light.direction), 0.0, 1.0);
        color += material_data.color.xyz * n_dot_l * light.color * light.attenuation;
    }

    out_color = vec4(color, 1.0);
}
