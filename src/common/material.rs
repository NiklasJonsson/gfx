use std::path::PathBuf;

use trekanten::material::MaterialData;

use specs::prelude::*;
use specs::Component;

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
            _ => false,
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

    unimplemented!();

    // render_graph::map(world, root, change_to_runtime);
}
*/
