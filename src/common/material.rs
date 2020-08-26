use super::render_graph;
use crate::asset::storage::Handle;
use crate::render::texture::Texture;

use std::path::PathBuf;

use specs::prelude::*;
use specs::Component;

use super::*;

#[derive(Clone, Copy, Debug)]
pub struct TextureUse {
    pub handle: Handle<Texture>,
    pub coord_set: u32,
}

#[derive(Clone, Debug)]
pub struct NormalMap {
    pub tex: TextureUse,
    pub scale: f32,
}

/// Compile time means that the material will be used to lookup what pre-compiled shader to use.
#[derive(Debug, Clone, Eq)]
pub enum ShaderUse {
    StaticInferredFromMaterial,
    RunTime { vs_path: PathBuf, fs_path: PathBuf },
}

impl PartialEq for ShaderUse {
    fn eq(&self, other: &Self) -> bool {
        use ShaderUse::*;
        match (self, other) {
            (StaticInferredFromMaterial, StaticInferredFromMaterial) => true,
            (RunTime { .. }, RunTime { .. }) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum MaterialData {
    Color {
        color: [f32; 4],
    },
    ColorTexture(Texture),
    GlTFPBR {
        base_color_factor: [f32; 4],
        metallic_factor: f32,
        roughness_factor: f32,
        normal_map: Option<NormalMap>,
        base_color_texture: Option<TextureUse>,
        metallic_roughness_texture: Option<TextureUse>,
    },
    None,
}

#[derive(Clone, Debug, Component)]
#[storage(DenseVecStorage)]
pub struct Material {
    pub data: MaterialData,
    pub compilation_mode: ShaderUse,
}

pub fn runtime_shaders_for_material(
    world: &World,
    root: Entity,
    vs_path: impl Into<PathBuf>,
    fs_path: impl Into<PathBuf>,
    match_material: impl Fn(&Material) -> bool,
) {
    let mut materials = world.write_storage::<Material>();
    let vs_path = vs_path.into();
    let fs_path = fs_path.into();
    let change_to_runtime = |ent| {
        if let Some(mat) = materials.get_mut(ent) {
            if match_material(&mat) {
                let vs_path = vs_path.clone();
                let fs_path = fs_path.clone();
                (*mat).compilation_mode = ShaderUse::RunTime { vs_path, fs_path };
            }
        }
    };

    render_graph::map(world, root, change_to_runtime);
}