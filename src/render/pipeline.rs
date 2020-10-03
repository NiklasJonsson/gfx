// TODO: Platform independent paths (look at ash)
macro_rules! GENERATED_BASE_PATH {
    () => {
        concat!(env!("OUT_DIR"), "/generated")
    };
}

macro_rules! GENERATED_CODE_PATH {
    () => {
        concat!(GENERATED_BASE_PATH!(), "/code")
    };
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShaderDefinition {
    pub has_tex_coords: bool,
    pub has_vertex_colors: bool,
    pub has_tangents: bool,
    pub has_base_color_texture: bool,
    pub has_metallic_roughness_texture: bool,
    pub has_normal_map: bool,
}

mod pbr_gltf {
    use super::ShaderDefinition;
    include! {concat!(GENERATED_CODE_PATH!(), "/gltf_pbr_shader_mapping.rs")}
}

pub struct PrecompiledShaders {
    vertex_shaders: std::collections::HashMap<ShaderDefinition, &'static str>,
    fragment_shaders: std::collections::HashMap<ShaderDefinition, &'static str>,
}

impl Default for PrecompiledShaders {
    fn default() -> Self {
        Self::new()
    }
}

impl PrecompiledShaders {
    pub fn new() -> Self {
        Self {
            vertex_shaders: pbr_gltf::vert_shader_mapping(),
            fragment_shaders: pbr_gltf::frag_shader_mapping(),
        }
    }

    pub fn get_vert(&self, d: &ShaderDefinition) -> &str {
        self.vertex_shaders
            .get(d)
            .expect("Missing precompiled shader")
    }

    pub fn get_frag(&self, d: &ShaderDefinition) -> &str {
        self.fragment_shaders
            .get(d)
            .expect("Missing precompiled shader")
    }

    pub fn get_default(&self) -> (&str, &str) {
        let def = ShaderDefinition::default();
        let vert = self.get_vert(&def);
        let frag = self.get_frag(&def);
        (vert, frag)
    }
}
