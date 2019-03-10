use crate::common::*;
use std::path::Path;

#[derive(PartialEq, Debug)]
pub enum AssetType {
    Static,
    Movable,
}

pub struct Asset {
    pub index_data: Vec<u32>,
    pub vertex_data: Vec<Vertex>,
    pub texture_data: image::RgbaImage,
    pub ty: AssetType,
}

pub struct AssetDescriptor {
    pub data_file: String,
    pub texture_file: String,
}

pub fn load_asset(descr: AssetDescriptor) -> Asset {
    let (vertex_data, index_data) = load_obj(&descr.data_file);
    let image = load_image(&descr.texture_file);

    Asset {
        index_data,
        vertex_data,
        texture_data: image,
        ty: AssetType::Static,
    }
}

fn debug_obj(models: &[tobj::Model], materials: &[tobj::Material]) {
    for (i, m) in models.iter().enumerate() {
        let mesh = &m.mesh;
        log::debug!("model[{}].name = \'{}\'", i, m.name);
        log::debug!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

        log::debug!("Size of model[{}].indices: {}", i, mesh.indices.len());
        /*
        for f in 0..mesh.indices.len() / 3 {
        log::debug!("    idx[{}] = {}, {}, {}.", f, mesh.indices[3 * f],
        mesh.indices[3 * f + 1], mesh.indices[3 * f + 2]);
        }
        */

        // Normals and texture coordinates are also loaded, but not printed in this example
        log::debug!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
        assert!(mesh.positions.len() % 3 == 0);
        /*
        for v in 0..mesh.positions.len() / 3 {
        log::debug!("    v[{}] = ({}, {}, {})", v, mesh.positions[3 * v],
        mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]);
        }
        */

        for (i, m) in materials.iter().enumerate() {
            log::debug!("material[{}].name = \'{}\'", i, m.name);
            log::debug!(
                "    material.Ka = ({}, {}, {})",
                m.ambient[0],
                m.ambient[1],
                m.ambient[2]
            );
            log::debug!(
                "    material.Kd = ({}, {}, {})",
                m.diffuse[0],
                m.diffuse[1],
                m.diffuse[2]
            );
            log::debug!(
                "    material.Ks = ({}, {}, {})",
                m.specular[0],
                m.specular[1],
                m.specular[2]
            );
            log::debug!("    material.Ns = {}", m.shininess);
            log::debug!("    material.d = {}", m.dissolve);
            log::debug!("    material.map_Ka = {}", m.ambient_texture);
            log::debug!("    material.map_Kd = {}", m.diffuse_texture);
            log::debug!("    material.map_Ks = {}", m.specular_texture);
            log::debug!("    material.map_Ns = {}", m.normal_texture);
            log::debug!("    material.map_d = {}", m.dissolve_texture);
            for (k, v) in &m.unknown_param {
                log::debug!("    material.{} = {}", k, v);
            }
        }
    }
}

fn load_obj(path: &str) -> (Vec<Vertex>, Vec<u32>) {
    log::info!("Loading models from {}", path);
    // TODO: "Vertex dedup" instead of storing duplicates of
    // vertices, we should re-use old ones if they are identical.

    let (models, materials) = tobj::load_obj(&Path::new(path)).unwrap();
    log::info!(
        "Found {} models and {} materials",
        models.len(),
        materials.len()
    );
    log::warn!("Ignoring materials and models other than model[0]");
    debug_obj(models.as_slice(), materials.as_slice());

    let tex_coords = models[0].mesh.texcoords.chunks_exact(2);
    let vertices = models[0]
        .mesh
        .positions
        .chunks_exact(3)
        .zip(tex_coords)
        .map(|(pos, tx_cs)| Vertex::new([pos[0], pos[1], pos[2]], [tx_cs[0], 1.0 - tx_cs[1]]))
        .collect::<Vec<_>>();

    let indices = models[0].mesh.indices.to_owned();

    (vertices, indices)
}

fn load_image(path: &str) -> image::RgbaImage {
    log::info!("Trying to load image from {}", path);
    let image = image::open(path).expect("Unable to load image").to_rgba();

    log::info!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    image
}
