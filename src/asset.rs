use crate::common::render_graph;
use crate::common::*;
use specs::prelude::*;
use specs::{Entity, World};
use std::path::{Path, PathBuf};
use vulkano::pipeline::input_assembly::PrimitiveTopology;

use gltf::buffer::Data as GltfData;

// Per asset type description, generally all the files needed to load an asset
pub enum AssetDescriptor {
    Obj {
        data_file: PathBuf,
        texture_file: PathBuf,
    },
    Gltf {
        path: PathBuf,
    },
}

pub struct LoadedAsset {
    pub scene_roots: Vec<Entity>,
    pub camera: Option<Transform>,
}

pub fn load_asset_into(world: &mut World, descr: AssetDescriptor) -> LoadedAsset {
    match descr {
        // TODO: Re-enable support
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

fn load_texture_common(
    ctx: &RecGltfCtx,
    texture: &gltf::texture::Texture,
    coord_set: u32,
    color_space: ColorSpace,
) -> Texture {
    assert_eq!(coord_set, 0, "Not implemented!");
    assert_eq!(
        texture.sampler().wrap_s(),
        gltf::texture::WrappingMode::Repeat
    );
    assert_eq!(
        texture.sampler().wrap_t(),
        gltf::texture::WrappingMode::Repeat
    );

    let image_src = texture.source().source();

    use gltf::image::Source;
    let image = match image_src {
        Source::Uri { uri, .. } => {
            let parent_path = Path::new(&ctx.path).parent().expect("Invalid path");
            let mut image_path = parent_path.to_path_buf();
            image_path.push(uri);
            load_image(image_path.to_str().expect("Could not create image path!"))
        }
        _ => unimplemented!(),
    };

    let format = Format {
        component_layout: ComponentLayout::R8G8B8A8,
        color_space,
    };

    Texture {
        image,
        coord_set,
        format,
    }
}

fn load_texture(ctx: &RecGltfCtx, texture_info: &gltf::texture::Info, cs: ColorSpace) -> Texture {
    load_texture_common(ctx, &texture_info.texture(), texture_info.tex_coord(), cs)
}

fn load_normal_map(ctx: &RecGltfCtx, normal_tex: &gltf::material::NormalTexture) -> NormalMap {
    // The normal map is always linear
    let tex = load_texture_common(
        ctx,
        &normal_tex.texture(),
        normal_tex.tex_coord(),
        ColorSpace::Linear,
    );
    let scale = normal_tex.scale();

    NormalMap { tex, scale }
}

fn get_primitives_from_mesh<'a>(ctx: &RecGltfCtx, mesh: gltf::Mesh<'a>) -> Vec<PolygonMesh> {
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
            let tangents = reader.read_tangents();
            let it = positions.zip(normals);

            let vertex_data = match (colors, tex_coords, tangents) {
                (None, Some(tex_coords), Some(tangents)) => VertexBuf::UVTan(
                    tex_coords
                        .into_f32()
                        .zip(tangents)
                        .zip(it)
                        .map(|((uv, tan), (pos, nor))| (pos, nor, uv, tan).into())
                        .collect::<Vec<VertexUVTan>>(),
                ),
                (Some(colors), Some(tex_coords), None) => VertexBuf::UVCol(
                    tex_coords
                        .into_f32()
                        .zip(colors.into_rgba_f32())
                        .zip(it)
                        .map(|((uv, col), (pos, nor))| (pos, nor, uv, col).into())
                        .collect::<Vec<VertexUVCol>>(),
                ),
                (None, Some(tex_coords), None) => VertexBuf::UV(
                    tex_coords
                        .into_f32()
                        .zip(it)
                        .map(|(uv, (pos, nor))| (pos, nor, uv).into())
                        .collect::<Vec<VertexUV>>(),
                ),
                (None, None, None) => VertexBuf::Base(
                    it.map(|pos_nor| pos_nor.into())
                        .collect::<Vec<VertexBase>>(),
                ),
                _ => unimplemented!(),
            };

            let mat = primitive.material();

            let pbr_mr = mat.pbr_metallic_roughness();

            if let Some(_) = mat.emissive_texture() {
                // TODO: Support this
                log::error!("No support for emissive texture!");
                unimplemented!();
            }

            // TODO: Only the first three components are in sRGB, alpha is linear!
            let base_color_texture = pbr_mr
                .base_color_texture()
                .map(|texture_info| load_texture(ctx, &texture_info, ColorSpace::Srgb));

            let metallic_roughness_texture = pbr_mr
                .metallic_roughness_texture()
                .map(|info| load_texture(ctx, &info, ColorSpace::Linear));

            let normal_map = mat
                .normal_texture()
                .map(|normal_map| load_normal_map(ctx, &normal_map));

            let material = Material::GlTFPBR {
                base_color_factor: pbr_mr.base_color_factor(),
                metallic_factor: pbr_mr.metallic_factor(),
                roughness_factor: pbr_mr.roughness_factor(),
                base_color_texture,
                metallic_roughness_texture,
                normal_map,
            };

            let triangle_indices = IndexData(triangle_index_data);

            let line_indices = generate_line_list_from(&triangle_indices);

            let bounding_box = Some(BoundingBox {
                min: primitive.bounding_box().min.into(),
                max: primitive.bounding_box().max.into(),
            });

            let ty = MeshType::Triangle {
                triangle_indices,
                line_indices,
            };

            // Default to static compilation, this can be fixed later
            let compilation_mode = CompilationMode::CompileTime;
            PolygonMesh {
                ty,
                vertex_data,
                material,
                compilation_mode,
                bounding_box,
            }
        })
        .collect::<Vec<_>>()
}

struct RecGltfCtx {
    pub buffers: Vec<GltfData>,
    pub path: PathBuf,
}

struct AssetGraphResult {
    // The enitity that was created
    node: Entity,
    // An entity somewhere in the graph that has a camera
    camera: Option<Entity>,
}

fn build_asset_graph_common<'a>(
    ctx: &RecGltfCtx,
    world: &'a mut World,
    src: &gltf::Node<'a>,
) -> AssetGraphResult {
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

    // For each child, we want its entity and if it has a camera
    // attached to it.
    let (mut nodes, cameras): (Vec<_>, Vec<Option<_>>) = src
        .children()
        .map(|child| {
            let AssetGraphResult { node, camera } = build_asset_graph_rec(ctx, world, &child, node);
            (node, camera)
        })
        .unzip();

    children.append(&mut nodes);

    // Use this nodes camera if it has one.
    let this_node_cam = src.camera().map(|_| node);
    // Folding with or ensures we will get exactly one camera index. This
    // currently prefers the current node camera if there are several, but we don't expect
    // more than 1 camera per scene.
    let camera: Option<Entity> = cameras.iter().fold(this_node_cam, |acc, &x| acc.or(x));

    let mut nodes = world.write_storage::<render_graph::RenderGraphNode>();
    nodes
        .insert(node, render_graph::node(children))
        .expect("Could not insert render graph node!");

    AssetGraphResult { node, camera }
}

fn build_asset_graph_rec<'a>(
    ctx: &RecGltfCtx,
    world: &mut World,
    src: &gltf::Node<'a>,
    parent: Entity,
) -> AssetGraphResult {
    let result = build_asset_graph_common(ctx, world, src);

    let mut nodes = world.write_storage::<render_graph::RenderGraphChild>();
    nodes
        .insert(result.node, render_graph::child(parent))
        .expect("Could not insert render graph node!");
    result
}

fn build_asset_graph(
    ctx: &RecGltfCtx,
    world: &mut World,
    src_root: &gltf::Node,
) -> AssetGraphResult {
    let result = build_asset_graph_common(ctx, world, src_root);

    let mut roots = world.write_storage::<render_graph::RenderGraphRoot>();
    roots
        .insert(result.node, render_graph::root())
        .expect("Could not insert render graph root!");

    result
}

pub fn load_gltf_asset(world: &mut World, path: &Path) -> LoadedAsset {
    log::trace!("load gltf asset {}", path.display());
    let (gltf_doc, buffers, _images) = gltf::import(path).expect("Unable to import gltf");
    assert_eq!(gltf_doc.scenes().len(), 1);
    let rec_ctx = RecGltfCtx {
        buffers,
        path: path.into(),
    };

    // A scene may have several root nodes
    let nodes = gltf_doc.scenes().nth(0).expect("No scenes!").nodes();
    if gltf_doc.scenes().len() > 1 {
        log::warn!("More than one scene found, only displaying the first");
        log::warn!("Number of scenes: {}", gltf_doc.scenes().len());
    }
    let mut roots: Vec<Entity> = Vec::new();
    let mut camera_ent: Option<Entity> = None;
    for node in nodes {
        log::trace!("Root node {}", node.name().unwrap_or("node_no_name"));
        log::trace!("#children {}", node.children().len());

        let AssetGraphResult { node, camera } = build_asset_graph(&rec_ctx, world, &node);
        roots.push(node);
        camera_ent = camera;
    }

    let mut cam_transform: Option<Transform> = None;
    if gltf_doc.cameras().nth(0).is_some() {
        log::info!("Found camera in scene");
        if gltf_doc.cameras().len() > 1 {
            log::warn!("More than one camera found, only using the first");
            log::warn!("Number of cameras: {}", gltf_doc.cameras().len());
        }

        if let Some(cam) = camera_ent {
            log::info!("Found camera in scene graph.");
            log::info!("Concatenating transforms.");
            let path = render_graph::root_to_node_path(world, cam);
            let mut transform: glm::Mat4 = glm::identity::<f32, glm::U4>();
            for ent in path {
                let transforms = world.read_storage::<Transform>();
                if let Some(t) = transforms.get(ent) {
                    let t: glm::Mat4 = (*t).into();
                    transform *= t;
                }
            }
            cam_transform = Some(transform.into());
        } else {
            log::info!("Did not find camera in scene graph");
            log::info!("Scanning the nodes for one with a camera");
            for node in gltf_doc.nodes() {
                if node.camera().is_some() {
                    log::info!("Found transform for camera!");
                    cam_transform = Some(node.transform().matrix().into());
                }
            }
        }

        if let Some(t) = cam_transform {
            log::info!("Final camera transform: {:#?}", t);
        }
    };

    LoadedAsset {
        scene_roots: roots,
        camera: cam_transform,
    }
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
        .unwrap_or_else(|_| panic!("Unable to load image from {}", path))
        .to_rgba();

    log::info!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    image
}
