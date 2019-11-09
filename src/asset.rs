use crate::common::render_graph;
use crate::common::*;
use specs::prelude::*;
use specs::{Entity, World};
use std::path::{Path, PathBuf};
use vulkano::pipeline::input_assembly::PrimitiveTopology;

use gltf::buffer::Data as GltfData;

// TODO: Move from String/str to PathBuf/Path

// Per asset type description, generally all the files needed to load an asset
// TODO: Change to path
pub enum AssetDescriptor {
    Obj {
        data_file: String,
        texture_file: String,
    },
    Gltf {
        path: String,
    },
}

pub fn load_asset_into(world: &mut World, descr: AssetDescriptor) -> Vec<Entity> {
    match descr {
        /*
        AssetDescriptor::Obj {
            data_file,
            texture_file,
        } => load_obj_asset(&data_file, &texture_file),
        */
        AssetDescriptor::Gltf { path } => load_gltf_asset(world, &path),
        _ => unimplemented!(),
    }
}

fn generate_line_list_from(index_data: &IndexData) -> IndexData {
    let IndexData {
        rendering_mode,
        data: indices,
    } = index_data;
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

fn get_primitives_from_mesh<'a>(ctx: &RecGltfCtx, mesh: gltf::Mesh<'a>) -> Vec<GraphicsPrimitive> {
    mesh.primitives()
        .map(|primitive| {
            let reader = primitive.reader(|buffer| Some(&ctx.buffers[buffer.index()]));
            let positions = reader.read_positions().expect("Found no positions");
            let normals = reader.read_normals().expect("Found no normals");
            assert!(primitive.mode() == gltf::mesh::Mode::Triangles);

            let triangle_index_data = reader
                .read_indices()
                .expect("Found no indices")
                .into_u32()
                .collect::<Vec<_>>();

            // TODO: Don't convert all tex_coords to f32
            let tex_coords = reader.read_tex_coords(0);
            let colors = reader.read_colors(0);
            let it = positions.zip(normals);

            let vertex_data = match (colors, tex_coords) {
                (Some(colors), Some(tex_coords)) => VertexBuf::UVCol(
                    tex_coords
                        .into_f32()
                        .zip(colors.into_rgba_f32())
                        .zip(it)
                        .map(|((uv, col), (pos, nor))| (pos, nor, uv, col).into())
                        .collect::<Vec<VertexUVCol>>(),
                ),
                (None, Some(tex_coords)) => VertexBuf::UV(
                    tex_coords
                        .into_f32()
                        .zip(it)
                        .map(|(uv, (pos, nor))| (pos, nor, uv).into())
                        .collect::<Vec<VertexUV>>(),
                ),
                (None, None) => VertexBuf::Base(
                    it.map(|pos_nor| pos_nor.into())
                        .collect::<Vec<VertexBase>>(),
                ),
                (Some(_), None) => unimplemented!(),
            };

            let pbr_mr = primitive.material().pbr_metallic_roughness();

            let tex = pbr_mr.base_color_texture().map(|texture_info| {
                assert_eq!(texture_info.tex_coord(), 0, "Not implemented!");
                assert_eq!(
                    texture_info.texture().sampler().wrap_s(),
                    gltf::texture::WrappingMode::Repeat
                );
                assert_eq!(
                    texture_info.texture().sampler().wrap_t(),
                    gltf::texture::WrappingMode::Repeat
                );

                let image_src = texture_info.texture().source().source();

                use gltf::image::Source;
                let image = match image_src {
                    Source::Uri { uri, .. } => {
                        let parent_path = Path::new(&ctx.path).parent().expect("Invalid path");
                        assert!(parent_path.has_root());
                        let mut image_path = parent_path.to_path_buf();
                        image_path.push(uri);
                        load_image(image_path.to_str().expect("Could not create image path!"))
                    }
                    _ => unimplemented!(),
                };

                Texture {
                    image,
                    coord_set: texture_info.tex_coord(),
                }
            });

            let material = Material::GlTFPBRMaterial {
                base_color_factor: pbr_mr.base_color_factor(),
                metallic_factor: pbr_mr.metallic_factor(),
                roughness_factor: pbr_mr.roughness_factor(),
                base_color_texture: tex,
            };

            let triangle_indices = IndexData::triangle_list(triangle_index_data);

            let line_indices = generate_line_list_from(&triangle_indices);

            GraphicsPrimitive {
                triangle_indices,
                line_indices,
                vertex_data,
                material,
            }
        })
        .collect::<Vec<_>>()
}

struct RecGltfCtx {
    pub buffers: Vec<GltfData>,
    pub path: PathBuf,
}

fn build_asset_graph_common<'a>(
    ctx: &RecGltfCtx,
    world: &'a mut World,
    src: &gltf::Node<'a>,
) -> Entity {
    let node = world
        .create_entity()
        .with(Transform::from(src.transform().matrix()))
        .build();

    let mut children = src
        .mesh()
        .map(|mesh| {
            get_primitives_from_mesh(ctx, mesh)
                .into_iter()
                .map(|graphics_primitive| {
                    world
                        .create_entity()
                        .with(graphics_primitive)
                        .with(render_graph::leaf(node))
                        .build()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(Vec::new);

    children.append(
        &mut src
            .children()
            .map(|child| build_asset_graph_rec(ctx, world, &child, node))
            .collect::<Vec<_>>(),
    );

    let mut nodes = world.write_storage::<render_graph::RenderGraphNode>();
    nodes
        .insert(node, render_graph::node(children))
        .expect("Could not insert render graph node!");

    node
}

fn build_asset_graph_rec<'a>(
    ctx: &RecGltfCtx,
    world: &mut World,
    src: &gltf::Node<'a>,
    parent: Entity,
) -> Entity {
    let node = build_asset_graph_common(ctx, world, src);

    let mut nodes = world.write_storage::<render_graph::RenderGraphChild>();
    nodes
        .insert(node, render_graph::child( parent ))
        .expect("Could not insert render graph!");
    node
}

fn build_asset_graph(ctx: &RecGltfCtx, world: &mut World, src_root: &gltf::Node) -> Entity {
    let root = build_asset_graph_common(ctx, world, src_root);

    let mut roots = world.write_storage::<render_graph::RenderGraphRoot>();
    roots
        .insert(root, render_graph::root())
        .expect("Could not insert render graph root!");

     root
}

pub fn load_gltf_asset(world: &mut World, path: &str) -> Vec<Entity> {
    log::trace!("load gltf asset {}", path);
    let (gltf_doc, buffers, _images) = gltf::import(path).expect("Unable to import gltf");
    assert_eq!(gltf_doc.scenes().len(), 1);
    let rec_ctx = RecGltfCtx {
        buffers,
        path: path.into(),
    };

    // A scene may have several root nodes
    let nodes = gltf_doc.scenes().nth(0).expect("No scenes!").nodes();
    let mut roots: Vec<Entity> = Vec::new();
    for node in nodes {
        log::trace!("Root node {}", node.name().unwrap_or("node_no_name"));
        log::trace!("#children {}", node.children().len());

        roots.push(build_asset_graph(&rec_ctx, world, &node));
    }

    roots
}

/* TODO:
fn load_obj_asset(data_file: &str, texture_file: &str) -> Asset {
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

pub fn load_image(path: &str) -> image::RgbaImage {
    log::info!("Trying to load image from {}", path);
    let image = image::open(path)
        .expect(format!("Unable to load image from {}", path).as_str())
        .to_rgba();

    log::info!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    image
}
