use crate::common::*;
use std::path::Path;
use vulkano::pipeline::input_assembly::PrimitiveTopology;

pub struct IndexData {
    pub rendering_mode: PrimitiveTopology,
    pub data: Vec<u32>,
}

impl IndexData {
    pub fn line_list(data: Vec<u32>) -> Self {
        IndexData { rendering_mode: PrimitiveTopology::LineList, data }
    }

    pub fn triangle_list(data: Vec<u32>) -> Self {
        IndexData { rendering_mode: PrimitiveTopology::TriangleList, data }
    }
}

// One ore more vertices with associated data
// One transform per primitive.
pub struct Primitive {
    pub triangle_indices: IndexData,
    pub line_indices: IndexData,
    pub vertex_data: Vec<Vertex>,
    pub texture_data: Option<image::RgbaImage>,
    pub transform: Option<glm::Mat4>,
    pub color: Option<glm::Vec4>,
}

// Describes an asset.
// TODO: Support hierarchy
pub struct Asset {
    pub primitives: Vec<Primitive>
}

// Per asset type description, generally all the files needed to load an asset
pub enum AssetDescriptor {
    Obj{data_file: String, texture_file: String},
    Gltf{path: String},
}

pub fn load_asset(descr: AssetDescriptor) -> Asset {
    match descr {
        AssetDescriptor::Obj{data_file, texture_file} => {
            load_obj_asset(&data_file, &texture_file)
        },
        AssetDescriptor::Gltf{path} => {
            load_glTF_asset(&path)
        }
    }
}

fn generate_line_list_from(index_data: &IndexData) -> IndexData {
    let IndexData { rendering_mode, data: indices } = index_data;
    let mut ret = Vec::new();
    assert_eq!(*rendering_mode, PrimitiveTopology::TriangleList);
    assert_eq!(indices.len() % 3, 0);
    for triangle in indices.chunks(3) {
        ret.push(triangle[0]);
        ret.push(triangle[1]);
        ret.push(triangle[1]);
        ret.push(triangle[2]);
        ret.push(triangle[2]);
        ret.push(triangle[0]);
    }

    IndexData::line_list(ret)
}

fn gltf_to_vulkano_topology(mode: gltf::mesh::Mode) -> PrimitiveTopology {

    use gltf::mesh::Mode;

    match mode {
        Mode::Points => PrimitiveTopology::PointList,
	Mode::Lines => PrimitiveTopology::LineList,
	Mode::LineLoop => panic!("Not supported!?"),
	Mode::LineStrip => PrimitiveTopology::LineStrip,
	Mode::Triangles => PrimitiveTopology::TriangleList,
	Mode::TriangleStrip => PrimitiveTopology::TriangleStrip,
	Mode::TriangleFan => PrimitiveTopology::TriangleFan,
    }
}

// This assumes column major
fn arr4x4_to_mat4(arr: &[[f32; 4]; 4]) -> glm::Mat4 {
    let mut tmp = [0.0f32; 16];

    let mut i = 0;
    for col in arr {
        for v in col {
            tmp[i] = *v;
            i += 1;
        }
    }
    glm::make_mat4(&tmp)
}

pub fn load_glTF_asset(path: &str) -> Asset {
    log::trace!("load gltf asset {}", path);
    let (gltf_doc, buffers, images) = gltf::import(path).expect("Unable to import gltf");
    assert_eq!(gltf_doc.scenes().len(), 1);
    let mut primitives = Vec::new();
    for node in gltf_doc.nodes() {
        log::trace!("Prepping node {}", node.name().unwrap_or("node_no_name"));
        log::trace!("#children {}", node.children().len());
        let model = node.transform().matrix();

        for child_node in node.children() {
            let mesh = child_node.mesh().unwrap();
            log::trace!("Prepping mesh: {}", mesh.name().unwrap_or("mesh_no_name"));

            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions = reader.read_positions().expect("Found no positions");
                let normals = reader.read_normals().expect("Found no normals");
                assert!(primitive.mode() == gltf::mesh::Mode::Triangles);
                /*
                // TODO: Don't convert all tex_coords to f32
                let tex_coords = reader
                    .read_tex_coords(0)
                    .expect("Found no tex coords")
                    .into_f32();
                */

                let triangle_index_data = reader
                    .read_indices()
                    .expect("Found no indices")
                    .into_u32()
                    .collect::<Vec<_>>();

                let vertex_data = positions
                    .zip(normals)
                    .map(|(p, n)| Vertex {position: p, normal: n})
                    .collect::<Vec<_>>();

                let pbr_mr = primitive.material().pbr_metallic_roughness();
                let color = pbr_mr.base_color_factor();

                let model: glm::Mat4x4 = glm::identity();

                let triangle_indices = IndexData::triangle_list(triangle_index_data);

                let line_indices = generate_line_list_from(&triangle_indices);

                primitives.push(
                    Primitive {
                        triangle_indices,
                        line_indices,
                        vertex_data,
                        texture_data: None,
                        transform: Some(model.into()),
                        color: Some(color.into())
                    }
                );
            }
        }
    }
    Asset{ primitives }
}

fn debug_gltf(file: gltf::Document) {
    for scene in file.scenes() {
        log::trace!("Scenes[{}]: {}", scene.index(), scene.name().unwrap_or_else(|| ""));
        for node in scene.nodes() {
            dbg!(node.mesh());
        }
    }
}

fn load_obj_asset(data_file: &str, texture_file: &str) -> Asset {
    let (vertex_data, index_data) = load_obj(data_file);
    let image = load_image(texture_file);

    let triangle_indices = IndexData::triangle_list(index_data);

    let line_indices = generate_line_list_from(&triangle_indices);
    Asset {
        primitives: vec![
            Primitive {
                triangle_indices,
                line_indices,
                vertex_data,
                texture_data: Some(image),
                transform: None,
                color: None,
            },
        ]
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
        .map(|p| Vertex {position: [p[0],p[1],p[2]], normal: [0.0f32; 3]})
        .collect::<Vec<_>>();

    let indices = models[0].mesh.indices.to_owned();

    (vertices, indices)
}

pub fn load_image(path: &str) -> image::RgbaImage {
    log::info!("Trying to load image from {}", path);
    let image = image::open(path).expect("Unable to load image").to_rgba();

    log::info!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    image
}
