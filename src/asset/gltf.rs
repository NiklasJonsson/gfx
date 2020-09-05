use crate::common::render_graph;
use specs::Component;

use specs::prelude::*;
use specs::world::EntitiesRes;
use specs::{Entity, World};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::LoadedAsset;

use crate::common::{Material, ShaderUse, Transform};
use crate::render::uniform::PBRMaterialData;
use crate::render::Mesh;
use trekanten::mesh::{IndexBuffer, IndexBufferDescriptor, VertexBuffer, VertexBufferDescriptor};
use trekanten::resource::ResourceManager;
use trekanten::uniform::{UniformBuffer, UniformBufferDescriptor};
use trekanten::util;
use trekanten::vertex::VertexFormat;
use trekanten::BufferHandle;
use trekanten::Handle;

use nalgebra_glm as glm;
// Crate gltf
#[derive(Debug)]
struct GltfVertexBuffer {
    data: Vec<u8>,
    format: VertexFormat,
}

// TODO: More u32 below
#[derive(Debug)]
struct GltfIndexBufferHandle {
    start: usize,
    size: usize,
}

impl GltfIndexBufferHandle {
    fn as_gpu_handle(&self, h: Handle<IndexBuffer>) -> BufferHandle<IndexBuffer> {
        BufferHandle::from_buffer(
            h,
            (self.start * std::mem::size_of::<u32>()) as u32,
            (self.size * std::mem::size_of::<u32>()) as u32,
            std::mem::size_of::<u32>() as u32,
        )
    }
}

#[derive(Debug)]
struct GltfVertexBufferHandle {
    buf_idx: usize,
    start_vertex: u32,
    n_vertices: u32,
    vertex_size: u32,
}

impl GltfVertexBufferHandle {
    fn as_gpu_handle(&self, h: Handle<VertexBuffer>) -> BufferHandle<VertexBuffer> {
        BufferHandle::from_buffer(
            h,
            self.start_vertex * self.vertex_size,
            self.n_vertices * self.vertex_size,
            self.vertex_size,
        )
    }
}

#[derive(Debug)]
struct GltfMaterialBufferHandle {
    idx: usize,
}

impl GltfMaterialBufferHandle {
    fn as_gpu_handle(&self, h: Handle<UniformBuffer>) -> BufferHandle<UniformBuffer> {
        BufferHandle::from_buffer(
            h,
            (self.idx * std::mem::size_of::<PBRMaterialData>()) as u32,
            std::mem::size_of::<PBRMaterialData>() as u32,
            std::mem::size_of::<PBRMaterialData>() as u32,
        )
    }
}

#[derive(Debug)]
struct GltfMaterial {
    material: GltfMaterialBufferHandle,
    normal_map: Option<GltfNormalMap>,
    base_color: Option<GltfTexture>,
    metallic_roughness: Option<GltfTexture>,
    has_vertex_colors: bool,
}

#[derive(Component)]
#[storage(VecStorage)]
struct GltfModel {
    mat: GltfMaterial,
    index: GltfIndexBufferHandle,
    vertex: GltfVertexBufferHandle,
}

#[derive(Debug)]
struct GltfTexture {
    path: PathBuf,
    format: util::Format,
}

#[derive(Debug)]
struct GltfNormalMap {
    tex: GltfTexture,
    scale: f32,
}

struct RecGltfCtx<'a> {
    pub buffers: Vec<gltf::buffer::Data>,
    pub path: PathBuf,
    pub world: &'a mut World,
    pub all_index_buffers: Vec<u32>,
    pub vertex_buffers: Vec<GltfVertexBuffer>,
    pub format_to_idx: HashMap<VertexFormat, usize>,
    pub material_buffer: Vec<PBRMaterialData>,
}

impl<'a> RecGltfCtx<'a> {
    fn add_index_buffer(&mut self, mut v: Vec<u32>) -> GltfIndexBufferHandle {
        let handle = GltfIndexBufferHandle {
            start: self.all_index_buffers.len(),
            size: v.len(),
        };

        self.all_index_buffers.append(&mut v);

        handle
    }

    fn add_vertex_buffer(&mut self, v: GltfVertexBuffer) -> GltfVertexBufferHandle {
        use std::collections::hash_map::Entry;
        let GltfVertexBuffer { mut data, format } = v;
        let vertex_size = format.size();
        let n_vertices = data.len() as u32 / vertex_size;

        // TODO: Refactor with or_insert()?
        match self.format_to_idx.entry(format.clone()) {
            Entry::Occupied(entry) => {
                let idx = entry.get();
                let entry = &mut self.vertex_buffers[*idx];
                assert_eq!(entry.data.len() % entry.format.size() as usize, 0);
                assert_eq!(vertex_size, entry.format.size());

                let h = GltfVertexBufferHandle {
                    buf_idx: *idx,
                    start_vertex: entry.data.len() as u32 / vertex_size,
                    n_vertices,
                    vertex_size,
                };
                entry.data.append(&mut data);
                h
            }
            Entry::Vacant(entry) => {
                let h = GltfVertexBufferHandle {
                    buf_idx: self.vertex_buffers.len(),
                    start_vertex: 0,
                    n_vertices,
                    vertex_size,
                };
                entry.insert(self.vertex_buffers.len());
                self.vertex_buffers.push(GltfVertexBuffer { format, data });
                h
            }
        }
    }

    fn add_material_data(&mut self, data: PBRMaterialData) -> GltfMaterialBufferHandle {
        self.material_buffer.push(data);
        GltfMaterialBufferHandle {
            idx: self.material_buffer.len() - 1,
        }
    }
}

fn load_texture_common<'a>(
    ctx: &RecGltfCtx<'a>,
    texture: &gltf::texture::Texture,
    coord_set: u32,
    format: util::Format,
) -> GltfTexture {
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
        x => unimplemented!("Unsupported image source {:?}", x),
    };

    GltfTexture {
        path: image_path,
        format,
    }
}

fn load_texture(
    ctx: &RecGltfCtx,
    texture_info: &gltf::texture::Info,
    cs: util::Format,
) -> GltfTexture {
    load_texture_common(ctx, &texture_info.texture(), texture_info.tex_coord(), cs)
}

fn load_normal_map(ctx: &RecGltfCtx, normal_tex: &gltf::material::NormalTexture) -> GltfNormalMap {
    // The normal map is always linear
    let tex = load_texture_common(
        ctx,
        &normal_tex.texture(),
        normal_tex.tex_coord(),
        util::Format::RGBA_UNORM,
    );
    let scale = normal_tex.scale();

    GltfNormalMap { tex, scale }
}

fn check_supported<'a>(primitive: &gltf::Primitive<'a>) {
    use gltf::mesh::Semantic;
    for (semantic, _accessor) in primitive.attributes() {
        match semantic {
            Semantic::Positions => (),
            Semantic::Normals => (),
            Semantic::Tangents => (),
            Semantic::Colors(0) => (),
            Semantic::TexCoords(0) => (),
            _ => unimplemented!("Unsupported semantic: {:?}", semantic),
        }
    }
}

// TODO: Find a way to handle binding mapping here and in shader in one place.
fn interleave_vertex_buffer<'a>(
    ctx: &RecGltfCtx,
    primitive: &gltf::Primitive<'a>,
) -> (GltfVertexBuffer, bool) {
    check_supported(primitive);
    let reader = primitive.reader(|buffer| Some(&ctx.buffers[buffer.index()]));
    let positions = reader.read_positions().expect("Found no positions");
    let normals = reader.read_normals().expect("Found no normals");

    let mut format = VertexFormat::builder()
        .add_attribute(util::Format::FLOAT3) // position
        .add_attribute(util::Format::FLOAT3); // normal

    let tangents = reader.read_tangents();
    let tex_coords = reader.read_tex_coords(0);
    let colors = reader.read_colors(0);

    if tex_coords.is_some() {
        format = format.add_attribute(util::Format::FLOAT2);
    }

    if colors.is_some() {
        format = format.add_attribute(util::Format::FLOAT4);
    }

    if tangents.is_some() {
        format = format.add_attribute(util::Format::FLOAT4);
    }

    let format = format.build();

    // TODO: Prealloc
    let mut data = Vec::new();
    let has_vertex_colors = colors.is_some();

    // TODO: How to map this to shader layout?
    let it = positions.zip(normals);
    match (colors, tex_coords, tangents) {
        (None, Some(tex_coords), Some(tangents)) => {
            for ((uv, tan), (pos, nor)) in tex_coords.into_f32().zip(tangents).zip(it) {
                data.extend_from_slice(util::as_bytes(&pos));
                data.extend_from_slice(util::as_bytes(&nor));
                data.extend_from_slice(util::as_bytes(&uv));
                data.extend_from_slice(util::as_bytes(&tan));
            }
        }
        (Some(colors), Some(tex_coords), None) => {
            for ((uv, col), (pos, nor)) in tex_coords.into_f32().zip(colors.into_rgba_f32()).zip(it)
            {
                data.extend_from_slice(util::as_bytes(&pos));
                data.extend_from_slice(util::as_bytes(&nor));
                data.extend_from_slice(util::as_bytes(&uv));
                data.extend_from_slice(util::as_bytes(&col));
            }
        }
        (None, Some(tex_coords), None) => {
            for (uv, (pos, nor)) in tex_coords.into_f32().zip(it) {
                data.extend_from_slice(util::as_bytes(&pos));
                data.extend_from_slice(util::as_bytes(&nor));
                data.extend_from_slice(util::as_bytes(&uv));
            }
        }
        (None, None, None) => {
            for (pos, nor) in it {
                data.extend_from_slice(util::as_bytes(&pos));
                data.extend_from_slice(util::as_bytes(&nor));
            }
        }
        _ => unimplemented!("Unsupported vertex format"),
    }

    (GltfVertexBuffer { data, format }, has_vertex_colors)
}

fn load_primitive<'a>(ctx: &mut RecGltfCtx, primitive: &gltf::Primitive<'a>) -> GltfModel {
    assert!(primitive.mode() == gltf::mesh::Mode::Triangles);
    let reader = primitive.reader(|buffer| Some(&ctx.buffers[buffer.index()]));

    let triangle_index_data = reader
        .read_indices()
        .expect("Found no indices")
        .into_u32()
        .collect::<Vec<_>>();
    let index_handle = ctx.add_index_buffer(triangle_index_data);
    let (vbuf, has_vertex_colors) = interleave_vertex_buffer(ctx, primitive);
    let vertex_handle = ctx.add_vertex_buffer(vbuf);

    let mat = primitive.material();
    let pbr_mr = mat.pbr_metallic_roughness();
    if mat.emissive_texture().is_some() {
        unimplemented!("No support for emissive texture!");
    }

    let base_color_texture = pbr_mr
        .base_color_texture()
        .map(|texture_info| load_texture(ctx, &texture_info, util::Format::RGBA_SRGB));

    let metallic_roughness_texture = pbr_mr
        .metallic_roughness_texture()
        .map(|info| load_texture(ctx, &info, util::Format::RGBA_UNORM));

    let normal_map = mat
        .normal_texture()
        .map(|normal_map| load_normal_map(ctx, &normal_map));

    let material_data = PBRMaterialData {
        base_color_factor: pbr_mr.base_color_factor(),
        metallic_factor: pbr_mr.metallic_factor(),
        roughness_factor: pbr_mr.roughness_factor(),
        normal_scale: normal_map.as_ref().map(|x| x.scale).unwrap_or(1.0),
        _padding: 0.0,
    };

    let mat_handle = ctx.add_material_data(material_data);

    GltfModel {
        mat: GltfMaterial {
            material: mat_handle,
            normal_map,
            base_color: base_color_texture,
            metallic_roughness: metallic_roughness_texture,
            has_vertex_colors,
        },
        vertex: vertex_handle,
        index: index_handle,
    }
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
            mesh.primitives()
                .map(|primitive| {
                    let gltf_model = load_primitive(ctx, &primitive);
                    ctx.world
                        .create_entity()
                        .with(gltf_model)
                        .with(render_graph::leaf(node))
                        .build()
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(Vec::new);

    // For each child *node*, we want its entity and if it has a camera
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

fn get_cam_transform(
    gltf_doc: gltf::Document,
    world: &World,
    camera_ent: Option<Entity>,
) -> Option<Transform> {
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
    }
    cam_transform
}

fn log_asset_upload<'a>(ctx: &RecGltfCtx<'a>) {
    log::info!("Uploading gltf asset to gpu");
    log::info!("# vertex_buffers: {}", ctx.vertex_buffers.len());
    for (i, vb) in ctx.vertex_buffers.iter().enumerate() {
        log::info!("vertex buffer {}:", i);
        log::info!("format: {:#?}", vb.format);
        log::info!(
            "n vertices: {} ({} / {})",
            vb.data.len() / vb.format.size() as usize,
            vb.data.len(),
            vb.format.size()
        );
    }
    log::info!("index_buffer len: {}", ctx.all_index_buffers.len());
    log::info!("# materials: {}", ctx.material_buffer.len());
}

fn upload_to_gpu<'a>(renderer: &mut trekanten::Renderer, ctx: &mut RecGltfCtx<'a>) {
    let mut meshes = ctx.world.write_storage::<Mesh>();
    let mut materials = ctx.world.write_storage::<Material>();
    let mut gltf_models = ctx.world.write_storage::<GltfModel>();
    let entities = ctx.world.read_resource::<EntitiesRes>();

    log_asset_upload(ctx);

    let gpu_vert_buffers: Vec<Handle<VertexBuffer>> = ctx
        .vertex_buffers
        .iter()
        .map(|vert_buf| {
            renderer
                .create_resource(VertexBufferDescriptor::from_raw(
                    &vert_buf.data,
                    vert_buf.format.clone(),
                ))
                .expect("Failed to create vertex buffer")
        })
        .collect();

    let gpu_index_buffer: Handle<IndexBuffer> = renderer
        .create_resource(IndexBufferDescriptor::from_slice(&ctx.all_index_buffers))
        .expect("Failed to create index buffer");

    let gpu_uniform_buffer = renderer
        .create_resource(UniformBufferDescriptor::from_slice(&ctx.material_buffer))
        .expect("Failed to create uniform buffer for materials");

    for (ent, model) in (&entities, &gltf_models).join() {
        let gltf_vh = &model.vertex;
        let gltf_ih = &model.index;
        let gltf_mat = &model.mat;
        let gpu_vh = gpu_vert_buffers[gltf_vh.buf_idx];

        let vertex_buffer = gltf_vh.as_gpu_handle(gpu_vh);
        let index_buffer = gltf_ih.as_gpu_handle(gpu_index_buffer);
        meshes
            .insert(
                ent,
                Mesh(trekanten::mesh::Mesh {
                    vertex_buffer,
                    index_buffer,
                }),
            )
            .expect("Failed to insert mesh");

        let material_uniforms = gltf_mat.material.as_gpu_handle(gpu_uniform_buffer);
        let normal_map = gltf_mat.normal_map.as_ref().map(|x| {
            let tex_h = renderer
                .create_resource(trekanten::texture::TextureDescriptor::new(
                    x.tex.path.clone(),
                    x.tex.format,
                ))
                .expect("Failed to create texture handle for normal map");

            let tex_use = trekanten::material::TextureUse {
                handle: tex_h,
                coord_set: 0,
            };

            trekanten::material::NormalMap {
                tex: tex_use,
                scale: x.scale,
            }
        });

        let base_color_texture = gltf_mat.base_color.as_ref().map(|t| {
            let tex_h = renderer
                .create_resource(trekanten::texture::TextureDescriptor::new(
                    t.path.clone(),
                    t.format,
                ))
                .expect("Failed to create texture handle for normal map");
            trekanten::material::TextureUse {
                handle: tex_h,
                coord_set: 0,
            }
        });

        let metallic_roughness_texture = gltf_mat.metallic_roughness.as_ref().map(|t| {
            let tex_h = renderer
                .create_resource(trekanten::texture::TextureDescriptor::new(
                    t.path.clone(),
                    t.format,
                ))
                .expect("Failed to create texture handle for normal map");
            trekanten::material::TextureUse {
                handle: tex_h,
                coord_set: 0,
            }
        });
        let mat_data = trekanten::material::MaterialData::PBR {
            material_uniforms,
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
            has_vertex_colors: gltf_mat.has_vertex_colors,
        };

        materials
            .insert(
                ent,
                Material {
                    data: mat_data,
                    compilation_mode: ShaderUse::PreCompiled,
                },
            )
            .expect("Failed to insert material");
    }

    gltf_models.clear();
}

pub fn load_asset(
    world: &mut World,
    renderer: &mut trekanten::Renderer,
    path: &Path,
) -> LoadedAsset {
    log::trace!("load gltf asset {}", path.display());
    let (gltf_doc, buffers, _images) = gltf::import(path).expect("Unable to import gltf");
    assert_eq!(gltf_doc.scenes().len(), 1);
    let mut rec_ctx = RecGltfCtx {
        buffers,
        path: path.into(),
        world,
        all_index_buffers: Vec::new(),
        vertex_buffers: Vec::new(),
        format_to_idx: HashMap::new(),
        material_buffer: Vec::new(),
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
        log::trace!("# children {}", node.children().len());

        let AssetGraphResult { node, camera } = build_asset_graph(&mut rec_ctx, &node);
        roots.push(node);
        camera_ent = camera;
    }

    let cam_transform = get_cam_transform(gltf_doc, rec_ctx.world, camera_ent);

    upload_to_gpu(renderer, &mut rec_ctx);

    LoadedAsset {
        scene_roots: roots,
        camera: cam_transform,
    }
}

pub fn register_components(world: &mut World) {
    world.register::<GltfModel>();
}
