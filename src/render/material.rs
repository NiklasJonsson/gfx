use trekanten::texture::Texture;
use trekanten::uniform::UniformBuffer;
use trekanten::BufferHandle;
use trekanten::Handle;

use specs::Component;
use specs::DenseVecStorage;

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TextureUse {
    pub handle: Handle<Texture>,
    pub coord_set: u32,
}

// TODO: Do we need to pass around the scale here?
#[derive(Debug, Clone)]
pub struct NormalMap {
    pub tex: TextureUse,
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub enum MaterialData {
    UniformColor {
        color: [f32; 4],
    },
    PBR {
        material_uniforms: BufferHandle<UniformBuffer>,
        normal_map: Option<NormalMap>,
        base_color_texture: Option<TextureUse>,
        metallic_roughness_texture: Option<TextureUse>,
        has_vertex_colors: bool,
    },
}

/// PreCompiled means that the material will be used to lookup what pre-compiled shader to use.
#[derive(Debug, Clone, Eq)]
pub enum ShaderUse {
    PreCompiled,
    Reloadable { vs_path: PathBuf, fs_path: PathBuf },
}

impl PartialEq for ShaderUse {
    fn eq(&self, other: &Self) -> bool {
        use ShaderUse::*;
        match (self, other) {
            (PreCompiled, PreCompiled) => true,
            (Reloadable { .. }, Reloadable { .. }) => true,
            (PreCompiled, Reloadable { .. }) => false,
            (Reloadable { .. }, PreCompiled) => false,
        }
    }
}

#[derive(Debug, Clone, Component)]
#[storage(DenseVecStorage)]
pub struct Material {
    pub data: MaterialData,
    pub compilation_mode: ShaderUse,
}

/*
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

    // transform_graph::map(world, root, change_to_runtime);
}
*/
