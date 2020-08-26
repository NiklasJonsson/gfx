#version 450
#extension GL_ARB_separate_shader_objects : enable

// Implementation is from Real-time Rendering, 4th edition

#define M_PI (3.1415926535897932384626433832795)

layout(set = 2, binding = 0) uniform LightingData {
    vec4 light_pos;
    vec4 view_pos;
} lighting_data;

layout(location = 0) in vec3 world_normal;
layout(location = 1) in vec3 world_pos;
#if HAS_TEX_COORDS
layout(location = TEX_COORDS_LOC) in vec2 tex_coords_0;
#endif

#if HAS_VERTEX_COLOR
layout(location = VCOL_LOC) in vec3 color_0;
#endif

#if HAS_TANGENTS
layout(location = TAN_LOC) in vec3 world_tangent;
layout(location = BITAN_LOC) in vec3 world_bitangent;
#endif

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

vec3 fresnel(vec3 fresnel_0, float VdotH) {
    vec3 base = vec3(1.0) - VdotH;
    return fresnel_0 + (1.0 - fresnel_0) * pow(base, vec3(5.0));
}

// GGX / Trowbridge-Reitz
float normal_distribution_function(float NdotH, float alpha_roughness) {
    float a2 = pow(alpha_roughness, 2.0);
    float bottom = M_PI * pow(1.0 + pow(NdotH, 2.0) * (a2 - 1.0), 2.0);

    return a2 / bottom;
}

void main() {
    vec3 normal = normalize(world_normal);

#if HAS_NORMAL_MAP
    vec3 tangent = normalize(world_tangent);
    vec3 bitangent = normalize(world_bitangent);
    mat3 tbn = mat3(tangent, bitangent, normal);
    vec3 tex_normal = texture(normal_map, tex_coords_0).xyz * 2.0 - 1.0;
    // Only scale .xy, as per the gltf spec
    tex_normal *= vec3(material_data.normal_scale, material_data.normal_scale, 1.0);
    normal = normalize(tbn * tex_normal);
#endif

    vec3 light_dir = normalize(lighting_data.light_pos.xyz - world_pos);
    vec3 view_dir = normalize(lighting_data.view_pos.xyz - world_pos);
    vec3 bisect_light_view = normalize(view_dir + light_dir);

    // NdotL is frequently used to determine if light can directly hit this point
    float NdotL = clamp(dot(normal, light_dir), 0.0, 1.0);
    float NdotV = clamp(dot(normal, view_dir), 0.0, 1.0);
    float NdotH = clamp(dot(normal, bisect_light_view), 0.0, 1.0);
    float VdotH = clamp(dot(view_dir, bisect_light_view), 0.0, 1.0);

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

    // TODO: Support textures for these
    vec3 base_color = material_data.base_color_factor.xyz;
    float metallic = material_data.metallic_factor;
    float roughness = material_data.roughness_factor;

    // For non-metals, we assume a F0 of 0.04. In reality it varies between 2-5% but we simplify
    vec3 dielectric_specular = vec3(0.04);
    vec3 black = vec3(0.0);

#if HAS_BASE_COLOR_TEXTURE
    base_color *= texture(base_color_texture, tex_coords_0).rgb;
#endif

#if HAS_METALLIC_ROUGHNESS_TEXTURE
    vec4 mr_tex = texture(metallic_roughness_texture, tex_coords_0);
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
    vec3 diffuse_color = mix(base_color * (1.0 - dielectric_specular.x), black, metallic);

    // The fresnel effect is essentially larger viewing angle => larger specular reflection of light
    // fresnel_0 is the factor of the incoming light that is *reflected* at 0 degrees
    // Refracted is 1 - fresnel_0 (conservation of energy - no light/energy is created)

    // With this interpolation we say that metallic materials get their color from base_color,
    // used later as specular reflection! Whilst non-metals get almost no color.
    vec3 fresnel_0 = mix(dielectric_specular, base_color, metallic);

    // Not to be confused with the color alpha/transparency
    float alpha_roughness = pow(roughness, 2.0);

    // Up until now, specific to gltf. From here on, wild west brdf
    // Define output as diffuse term + specular term
    vec3 fresnel = NdotL > 0.0 ? fresnel(fresnel_0, VdotH) : black;

    // diffuse term is the result of subsurface scattering, model as lambertian
    // term modified by the amount of light refracted (later in mix())
    vec3 diffuse = diffuse_color / M_PI;

	// Diffuse and specular both depend on cos between the normal and light vectors.
	// If the light is modeled as rays, the distance between the points where the light
	// rays hit the surface decreases (=> light intensity increases) as the light vector
	// approaches the normal.
	diffuse *= NdotL;

    // specular term is microfacet based, assuming each micro-facet is fresnel mirror
    // *Very* unoptimized version: (F * G * D) / ( 4 * dot(n,l) * dot(n,v))
    // If we choose GGX/Trowbridge-Reitz for normal distribution function (D) and
    // the Smith height-corelated masking-shadowing function for G, we can rewrite it
    // (see Real-time rendering 4th ed. for details)

    vec3 specular = vec3(0.0);
    // Only do this if the Light can hit the point
    if (NdotL > 0.0) {
        float divisor_0 = NdotV * sqrt(pow(alpha_roughness, 2.0) + pow(NdotL, 2.0) - pow(alpha_roughness * NdotL, 2.0));
        float divisor_1 = NdotL * sqrt(pow(alpha_roughness, 2.0) + pow(NdotV, 2.0) - pow(alpha_roughness * NdotV, 2.0));
        float OptTerm = 0.5/(divisor_0 + divisor_1);
        vec3 D = vec3(normal_distribution_function(NdotH, alpha_roughness));

        specular = OptTerm * D;
    }

    vec3 color = mix(diffuse, specular, fresnel);

    // TODO:
    // - More variations on BRDF
    // - lighting

    out_color = vec4(color, 1.0);
}
