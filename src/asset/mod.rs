use crate::common::math;
use crate::common::Format;
use crate::common::IndexData;
use specs::{Entity, World};
use std::path::PathBuf;

pub mod cache;
mod gltf;
mod obj;
pub mod storage;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    pub path: PathBuf,
    pub format: Format,
}

// Per asset type description, generally all the files needed to load an asset
#[derive(Debug)]
pub enum AssetDescriptor {
    Obj {
        data_file: PathBuf,
        texture_file: PathBuf,
    },
    Gltf {
        path: PathBuf,
    },
    Texture(TextureDescriptor),
}

pub struct LoadedAsset {
    pub scene_roots: Vec<Entity>,
    pub camera: Option<math::Transform>,
}

pub fn load_asset_into(world: &mut World, descr: AssetDescriptor) -> LoadedAsset {
    match descr {
        AssetDescriptor::Obj {
            data_file,
            texture_file,
        } => obj::load_asset(&data_file, &texture_file),
        AssetDescriptor::Gltf { path } => gltf::load_asset(world, &path),
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

pub fn load_image(desc: &TextureDescriptor) -> image::RgbaImage {
    let path = desc.path.to_str().expect("Failed to create path");

    log::info!("Trying to load image from {}", path);
    let image = image::open(path)
        .unwrap_or_else(|_| panic!("Unable to load image from {}", path))
        .to_rgba();

    log::info!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    image
}
