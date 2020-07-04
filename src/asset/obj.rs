use super::LoadedAsset;
use std::path::Path;

pub fn load_asset(data_file: &Path, texture_file: &Path) -> LoadedAsset {
    unimplemented!()
    /*
    let (vertex_data, index_data) = load_obj(data_file);
    let image = load_image(texture_file);

    let triangle_indices = IndexData::triangle_list(index_data);

    let line_indices = generate_line_list_from(&triangle_indices);

    let tex = Texture {
        image,
        coord_set: 0,
    };

    let material = Material::ColorTexture(tex);

    Asset {
        primitives: vec![Primitive {
            triangle_indices,
            line_indices,
            vertex_data,
            transform: None,
            material,
        }],
    }
    */
}

/*
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

fn load_obj(path: &str) -> (VertexBuf, Vec<u32>) {
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

    // TODO: Use these
    let tex_coords = models[0].mesh.texcoords.chunks_exact(2);
    let vertices = models[0]
        .mesh
        .positions
        .chunks_exact(3)
        .map(|p| VertexBase {
            position: [p[0], p[1], p[2]],
            normal: [0.0f32; 3],
        })
        .collect::<Vec<_>>();

    let indices = models[0].mesh.indices.to_owned();

    let vbuf = VertexBuf::Base(vertices);

    (vbuf, indices)
}
*/
