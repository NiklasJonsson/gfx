use crate::common::render_graph;
use crate::common::*;
use specs::prelude::*;
use specs::{Entity, World};
use std::path::{Path, PathBuf};

use super::generate_line_list_from;
use super::load_image;
use super::LoadedAsset;
use crate::render::texture::Texture;
use crate::render::texture::Textures;

use nalgebra_glm as glm;
// Crate gltf
use gltf::buffer::Data as GltfData;

struct RecGltfCtx<'a> {
    pub buffers: Vec<GltfData>,
    pub path: PathBuf,
    pub world: &'a mut World,
}

fn load_texture_common<'a>(
    ctx: &RecGltfCtx<'a>,
    texture: &gltf::texture::Texture,
    coord_set: u32,
    color_space: ColorSpace,
) -> TextureUse {
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
    let image_path = match image_src {
        Source::Uri { uri, .. } => {
            let parent_path = Path::new(&ctx.path).parent().expect("Invalid path");
            let mut image_path = parent_path.to_path_buf();
            image_path.push(uri);
            image_path
        }
        _ => unimplemented!(),
    };

    let format = Format {
        component_layout: ComponentLayout::R8G8B8A8,
        color_space,
    };

    let desc = super::TextureDescriptor {
        path: image_path,
        format,
    };

    let handle = ctx.world.write_resource::<Textures>().load(&desc);

    TextureUse { handle, coord_set }
}

fn load_texture(
    ctx: &RecGltfCtx,
    texture_info: &gltf::texture::Info,
    cs: ColorSpace,
) -> TextureUse {
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

fn get_primitives_from_mesh<'a>(
    ctx: &RecGltfCtx,
    gltf_mesh: gltf::Mesh<'a>,
) -> Vec<(Mesh, Material)> {
    gltf_mesh
        .primitives()
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

            if mat.emissive_texture().is_some() {
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

            let material_data = MaterialData::GlTFPBR {
                base_color_factor: pbr_mr.base_color_factor(),
                metallic_factor: pbr_mr.metallic_factor(),
                roughness_factor: pbr_mr.roughness_factor(),
                base_color_texture,
                metallic_roughness_texture,
                normal_map,
            };

            let material = Material {
                data: material_data,
                compilation_mode: ShaderUse::StaticInferredFromMaterial,
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
            let mesh = Mesh {
                ty,
                vertex_data,
                bounding_box,
            };

            (mesh, material)
        })
        .collect::<Vec<_>>()
}

struct AssetGraphResult {
    // The enitity that was created
    node: Entity,
    // An entity somewhere in the graph that has a camera
    camera: Option<Entity>,
}

fn build_asset_graph_common<'a>(ctx: &mut RecGltfCtx, src: &gltf::Node<'a>) -> AssetGraphResult {
    let node = ctx
        .world
        .create_entity()
        .with(Transform::from(src.transform().matrix()))
        .build();

    let mut children = src
        .mesh()
        .map(|mesh| {
            get_primitives_from_mesh(ctx, mesh)
                .into_iter()
                .map(|(mesh, material)| {
                    ctx.world
                        .create_entity()
                        .with(mesh)
                        .with(material)
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
            let AssetGraphResult { node, camera } = build_asset_graph_rec(ctx, &child, node);
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

    let mut nodes = ctx.world.write_storage::<render_graph::RenderGraphNode>();
    nodes
        .insert(node, render_graph::node(children))
        .expect("Could not insert render graph node!");

    AssetGraphResult { node, camera }
}

fn build_asset_graph_rec<'a>(
    ctx: &mut RecGltfCtx,
    src: &gltf::Node<'a>,
    parent: Entity,
) -> AssetGraphResult {
    let result = build_asset_graph_common(ctx, src);

    let mut nodes = ctx.world.write_storage::<render_graph::RenderGraphChild>();
    nodes
        .insert(result.node, render_graph::child(parent))
        .expect("Could not insert render graph node!");
    result
}

fn build_asset_graph(ctx: &mut RecGltfCtx, src_root: &gltf::Node) -> AssetGraphResult {
    let result = build_asset_graph_common(ctx, src_root);

    let mut roots = ctx.world.write_storage::<render_graph::RenderGraphRoot>();
    roots
        .insert(result.node, render_graph::root())
        .expect("Could not insert render graph root!");

    result
}

pub fn load_asset(world: &mut World, path: &Path) -> LoadedAsset {
    log::trace!("load gltf asset {}", path.display());
    let (gltf_doc, buffers, _images) = gltf::import(path).expect("Unable to import gltf");
    assert_eq!(gltf_doc.scenes().len(), 1);
    let mut rec_ctx = RecGltfCtx {
        buffers,
        path: path.into(),
        world,
    };

    // A scene may have several root nodes
    let nodes = gltf_doc.scenes().next().expect("No scenes!").nodes();
    if gltf_doc.scenes().len() > 1 {
        log::warn!("More than one scene found, only displaying the first");
        log::warn!("Number of scenes: {}", gltf_doc.scenes().len());
    }
    let mut roots: Vec<Entity> = Vec::new();
    let mut camera_ent: Option<Entity> = None;
    for node in nodes {
        log::trace!("Root node {}", node.name().unwrap_or("node_no_name"));
        log::trace!("#children {}", node.children().len());

        let AssetGraphResult { node, camera } = build_asset_graph(&mut rec_ctx, &node);
        roots.push(node);
        camera_ent = camera;
    }

    let mut cam_transform: Option<Transform> = None;
    if gltf_doc.cameras().next().is_some() {
        log::info!("Found camera in scene");
        if gltf_doc.cameras().len() > 1 {
            log::warn!("More than one camera found, only using the first");
            log::warn!("Number of cameras: {}", gltf_doc.cameras().len());
        }

        if let Some(cam) = camera_ent {
            log::info!("Found camera in scene graph.");
            log::info!("Concatenating transforms.");
            let path = render_graph::root_to_node_path(rec_ctx.world, cam);
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
