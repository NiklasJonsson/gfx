use crate::common::math;
use crate::common::IndexData;
use specs::{Entity, World};
use std::path::PathBuf;

mod gltf;

// Per asset type description, generally all the files needed to load an asset
#[derive(Debug)]
pub enum AssetDescriptor {
    Gltf { path: PathBuf },
}

pub struct LoadedAsset {
    pub scene_roots: Vec<Entity>,
    pub camera: Option<math::Transform>,
}

pub fn load_asset_into(
    world: &mut World,
    renderer: &mut trekanten::Renderer,
    descr: AssetDescriptor,
) -> LoadedAsset {
    match descr {
        AssetDescriptor::Gltf { path } => gltf::load_asset(world, renderer, &path),
        _ => unimplemented!(),
    }
}

fn generate_line_list_from(index_data: &IndexData) -> IndexData {
    let IndexData(indices) = index_data;
    let mut ret = Vec::new();
    assert_eq!(indices.len() % 3, 0);
    for triangle in indices.chunks(3) {
        ret.push(triangle[0]);
        ret.push(triangle[1]);
        ret.push(triangle[1]);
        ret.push(triangle[2]);
        ret.push(triangle[2]);
        ret.push(triangle[0]);
    }

    IndexData(ret)
}
