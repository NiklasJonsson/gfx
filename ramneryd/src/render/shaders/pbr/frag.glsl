#version 450
#extension GL_ARB_separate_shader_objects : enable

#include <engine.frag.glsl>

// Implementation is largely based on Real-time Rendering, 4th edition

layout(location = 0) in VsOut {
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

layout(location = 0) out vec4 out_color;

// PBR uniforms
// For base_color_factor, metallic_factor and roughness_factor, if there is not corresponding
// texture, each is used as the corresponding value in the computations below. If there is a
// texture, then each factor is multiplied with the sampled texture value
layout(set = 1, binding = 0) uniform PBRMaterialData {
    // If there is no base color texture, the base_color_factor is the color, otherwise
    // it is a multiplier for the texture values.
    vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
    // For the normal map, 1.0 if there is no map
    float normal_scale;
    float _padding;
} material_data;

#if HAS_BASE_COLOR_TEXTURE
layout(set = 1, binding = 1) uniform sampler2D base_color_texture;
#endif

#if HAS_METALLIC_ROUGHNESS_TEXTURE
layout(set = 1, binding = 2) uniform sampler2D metallic_roughness_texture;
#endif

#if HAS_NORMAL_MAP
layout(set = 1, binding = 3) uniform sampler2D normal_map;
#endif

// Schlick approx. cos_angle is the angle between the normal and the light.
vec3 fresnel(vec3 fresnel_0, float cos_angle) {
    return fresnel_0 + (vec3(1.0) - fresnel_0) * pow(1.0 - cos_angle, 5.0);
}

// GGX / Trowbridge-Reitz.
// cos_angle is cos of the angle between macro normal n and microfacet normal m (or h)
// In real-time renderering, m is used for the general case and h for specular reflection
float normal_distribution_function(float cos_angle, float alpha_roughness) {
    float a2 = pow(alpha_roughness, 2.0);
    float top = clamp(sign(cos_angle), 0.0, 1.0) * a2;
    float bottom = M_PI * pow(1.0 + pow(cos_angle, 2.0) * (a2 - 1.0), 2.0);

    return top / bottom;
}

#define DEBUG_SHADOW_MAP 0

vec3 debug_color(uint shadow_idx) {

#if DEBUG_SHADOW_MAP
    vec3 coords = vs_out.shadow_coords[shadow_idx].xyz / vs_out.shadow_coords[shadow_idx].w;

    if (coords.z > 1.0 || coords.z < -1.0) {
        return vec3(0.0);
    }

    float eps = 0.01;
    float map_size = 1024.0;
    vec2 real = map_size * coords.xy;
    bool x_close = abs(real.x - round(real.x)) < eps;
    bool y_close = abs(real.y - round(real.y)) < eps;
    if (x_close || y_close) {
        return vec3(0.5, 0.0, 0.0);
    }
#endif

    return vec3(0.0, 0.0, 0.0);
}

void main() {
    vec3 normal = normalize(vs_out.world_normal);

#if HAS_NORMAL_MAP
    vec3 tangent = normalize(vs_out.world_tangent);
    vec3 bitangent = normalize(vs_out.world_bitangent);
    mat3 tbn = mat3(tangent, bitangent, normal);
    vec3 tex_normal = texture(normal_map, vs_out.tex_coords_0).xyz * 2.0 - 1.0;
    // Only scale .xy, as per the gltf spec
    tex_normal *= vec3(material_data.normal_scale, material_data.normal_scale, 1.0);
    normal = normalize(tbn * tex_normal);
#endif

    /* ----------------- MATERIAL ------------------ */
    // Metallic-roughness/glTF PBR
    // Initial inputs come from textures and/or factors (uniforms)
    // These inputs are computed into inputs into the BRDF:
    //  * dielectric_specular
    //  * black
    //  * diffuse_color
    //  * fresnel_0
    //  * alpha
    // Explanations follow throughout the code
    // The choice of BRDF is not mandated by glTF

    vec3 base_color = material_data.base_color_factor.xyz;
    float metallic = material_data.metallic_factor;
    float roughness = material_data.roughness_factor;

    // For non-metals, we assume a F0 of 0.04. In reality it varies between 2-5% but we simplify
    float dielectric_specular = 0.04;
    vec3 black = vec3(0.0);

#if HAS_BASE_COLOR_TEXTURE
    base_color *= texture(base_color_texture, vs_out.tex_coords_0).rgb;
#endif

#if HAS_METALLIC_ROUGHNESS_TEXTURE
    vec4 mr_tex = texture(metallic_roughness_texture, vs_out.tex_coords_0);
    roughness *= mr_tex.g;
    metallic *= mr_tex.b;
#endif

#if HAS_VERTEX_COLOR
    base_color *= color_0.rgb;
#endif

    // Diffuse reflection is the light that is refracted into the material and then emitted again.
    // For most materials, the light is absorbed very quickly and we model this as a surface color
    // - diffuse_color.
    // The color of a material with a high metallic factor is black - a metal mostly reflects
    // and any tint is created with a specular reflection.
    vec3 diffuse_color = mix(base_color * (1.0 - dielectric_specular), black, metallic);

    // The fresnel effect is essentially larger viewing angle => larger specular reflection of light
    // fresnel_0 is the factor of the incoming light that is *reflected* at 0 degrees
    // Refracted is 1 - fresnel_0 (conservation of energy - no light/energy is created)

    // With this interpolation we say that metallic materials get their color from base_color,
    // used later as specular reflection! Whilst non-metals get almost no color.
    vec3 fresnel_0 = mix(vec3(dielectric_specular), base_color, metallic);

    // Not to be confused with the color alpha/transparency
    float alpha_roughness = pow(roughness, 2.0);

    /* ----------------- VIEW ------------------ */

    vec3 view_dir = normalize(view_data.view_pos.xyz - vs_out.world_pos);
    float n_dot_v = clamp(dot(normal, view_dir), 0.0, 1.0);

    /* ----------------- SHADING ------------------ */

    vec3 color = vec3(0);

    // Ambient
    color += lighting_data.ambient.xyz * diffuse_color * lighting_data.ambient.w;

    // Lights
    uint n_lights = num_lights();
    for (uint i = 0; i < n_lights; ++i) {
        Light light = unpack_light(lighting_data.lights[i], vs_out.world_pos);

        vec3 light_dir = light.direction;
        vec3 bisect_light_view = normalize(view_dir + light_dir);
        vec3 light_color = light.color;
        float attenuation = light.attenuation;

        float n_dot_l = clamp(dot(normal, light_dir), 0.0, 1.0);
        float n_dot_h_unclamped = dot(normal, bisect_light_view);
        float n_dot_h = clamp(n_dot_h_unclamped, 0.0, 1.0);
        float h_dot_l = clamp(dot(bisect_light_view, light_dir), 0.0, 1.0);

        color += debug_color(light.shadow_idx);

        float shadow_factor = compute_shadow_factor(vs_out.shadow_coords, light, n_dot_l);
        if (shadow_factor == 0.0) {
            continue;
        }

        // Define output as diffuse term + specular term
        vec3 fresnel = n_dot_l > 0.0 ? fresnel(fresnel_0, h_dot_l) : black;

        // diffuse factor is the result of subsurface scattering, model as lambertian
        // term modified by the amount of light refracted
        vec3 f_diffuse = (1.0 - fresnel) * diffuse_color / M_PI;

        // specular term is microfacet based, assuming each micro-facet is fresnel mirror
        // *Very* unoptimized version: (F * G * D) / ( 4 * dot(n,l) * dot(n,v))
        // If we choose GGX/Trowbridge-Reitz for normal distribution function (D) and
        // the Smith height-correlated masking-shadowing function for G, we can rewrite it
        // (see Real-time rendering 4th ed. for details)

        vec3 f_specular = vec3(0.0);
        // Only do this if the Light can hit the point
        if (n_dot_l > 0.0) {
            float a2 = pow(alpha_roughness, 2.0);
            float divisor_0 = n_dot_v * sqrt(a2 + n_dot_l * (n_dot_l - a2 * n_dot_l));
            float divisor_1 = n_dot_l * sqrt(a2 + n_dot_v * (n_dot_v - a2 * n_dot_v));
            float Vis = 0.5/(divisor_0 + divisor_1);
            vec3 D = vec3(normal_distribution_function(n_dot_h_unclamped, alpha_roughness));

            f_specular = Vis * D * fresnel;
        }

        // Diffuse and specular both depend on cos between the normal and light vectors.
        // If the light is modeled as rays, the distance between the points where the light
        // rays hit the surface decreases (=> light intensity increases) as the light vector
        // approaches the normal.
        color += (f_diffuse + f_specular) * n_dot_l * light_color * attenuation * shadow_factor;
    }
    // Punctual lights means the integral in the reflectance equation simplifies down to PI,
    // see real-time rendering 4, p. 316 eq. 9.14
    color *= M_PI;

    out_color = vec4(color, 1.0);
}
