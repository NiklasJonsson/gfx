use specs::Component;

use specs::prelude::*;
use specs::world::EntitiesRes;
use specs::{Entity, World};
use std::path::{Path, PathBuf};

use trekanten::mesh::BufferMutability;
use trekanten::mesh::{
    IndexBuffer, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor, VertexBuffer,
};
use trekanten::resource::ResourceManager;
use trekanten::texture::{MipMaps, TextureDescriptor};
use trekanten::uniform::OwningUniformBufferDescriptor;
use trekanten::util;
use trekanten::vertex::VertexFormat;
use trekanten::BufferHandle;

use super::LoadedAsset;

use crate::common::*;
use crate::ecs;
use crate::graph;
use crate::math::*;
use crate::render::material::Material;
use crate::render::uniform::PBRMaterialData;
use crate::render::Mesh;

// Crate gltf
#[derive(Debug)]
struct GltfVertexBuffer {
    data: Vec<u8>,
    format: VertexFormat,
}

#[derive(Debug)]
struct GltfIndexBufferHandle {
    buf_idx: usize,
    len: u32,
}

impl GltfIndexBufferHandle {
    fn as_gpu_handle(&self, h: BufferHandle<IndexBuffer>) -> BufferHandle<IndexBuffer> {
        BufferHandle::sub_buffer(h, 0, self.len)
    }
}

#[derive(Debug)]
struct GltfVertexBufferHandle {
    buf_idx: usize,
    n_vertices: u32,
    vertex_size: u32,
}

impl GltfVertexBufferHandle {
    fn as_gpu_handle(&self, h: BufferHandle<VertexBuffer>) -> BufferHandle<VertexBuffer> {
        BufferHandle::sub_buffer(h, 0, self.n_vertices)
    }
}

#[derive(Debug)]
struct GltfMaterialBufferHandle {
    buf_idx: usize,
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
    pub index_buffers: Vec<Vec<u32>>,
    pub vertex_buffers: Vec<GltfVertexBuffer>,
    pub material_buffer: Vec<PBRMaterialData>,
}

impl<'a> RecGltfCtx<'a> {
    fn add_index_buffer(&mut self, v: Vec<u32>) -> GltfIndexBufferHandle {
        let handle = GltfIndexBufferHandle {
            buf_idx: self.index_buffers.len(),
            len: v.len() as u32,
        };

        self.index_buffers.push(v);

        handle
    }

    fn add_vertex_buffer(&mut self, v: GltfVertexBuffer) -> GltfVertexBufferHandle {
        let GltfVertexBuffer { data, format } = v;
        let vertex_size = format.size();
        let n_vertices = data.len() as u32 / vertex_size;
        let h = GltfVertexBufferHandle {
            buf_idx: self.vertex_buffers.len(),
            n_vertices,
            vertex_size,
        };
        self.vertex_buffers.push(GltfVertexBuffer { format, data });

        h
    }

    fn add_material_data(&mut self, data: PBRMaterialData) -> GltfMaterialBufferHandle {
        self.material_buffer.push(data);
        GltfMaterialBufferHandle {
            buf_idx: self.material_buffer.len() - 1,
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

fn get_transform(src: gltf::scene::Transform) -> Transform {
    let (pos, rot, scale) = src.decomposed();
    let mut t = Transform::identity();
    t.position = Vec3::from(pos);
    t.rotation = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]).normalized();
    if !scale.iter().all(|x| *x == scale[0]) {
        log::warn!("Non-uniform scaling in asset: {:?}", scale);
        log::warn!("Using only {}", scale[0]);
    }
    t.scale = scale[0];

    t
}

fn load_node_rec(ctx: &mut RecGltfCtx, src: &gltf::Node) -> AssetGraphResult {
    let tfm = get_transform(src.transform());

    let node = ctx.world.create_entity().with(tfm).build();
    if let Some(name) = src.name() {
        ecs::assign(ctx.world, node, Name::from(name));
    }

    if let Some(mesh) = src.mesh() {
        for primitive in mesh.primitives() {
            let gltf_model = load_primitive(ctx, &primitive);
            let child = ctx.world.create_entity().with(gltf_model).build();

            graph::add_edge(ctx.world, node, child);
        }

        if let Some(name) = mesh.name() {
            let name = String::from(src.name().unwrap_or("")) + name;
            ecs::assign(ctx.world, node, Name::from(name));
        }
    }

    let mut camera = src.camera().map(|_| node);
    // For each child *node*, we want its entity and if it has a camera
    // attached to it.
    for gltf_child in src.children() {
        let AssetGraphResult {
            node: child,
            camera: child_camera,
        } = load_node_rec(ctx, &gltf_child);
        graph::add_edge(ctx.world, node, child);
        camera = camera.or(child_camera);
    }

    AssetGraphResult { node, camera }
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
            let path = graph::root_to_node_path(world, cam);
            let mut transform = Transform::identity();
            for ent in path {
                let transforms = world.read_storage::<Transform>();
                if let Some(t) = transforms.get(ent) {
                    transform *= *t;
                }
            }
            cam_transform = Some(transform);
        } else {
            log::info!("Did not find camera in scene graph");
            log::info!("Scanning the nodes for one with a camera");
            for node in gltf_doc.nodes() {
                if node.camera().is_some() {
                    log::info!("Found transform for camera!");
                    cam_transform = Some(get_transform(node.transform()));
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
        log::info!("format: {}", vb.format);
        log::info!(
            "n vertices: {} ({} / {})",
            vb.data.len() / vb.format.size() as usize,
            vb.data.len(),
            vb.format.size()
        );
    }
    log::info!("# index_buffers: {}", ctx.index_buffers.len());
    log::info!("# materials: {}", ctx.material_buffer.len());
}

fn upload_to_gpu<'a>(renderer: &mut trekanten::Renderer, ctx: RecGltfCtx<'a>) {
    log_asset_upload(&ctx);

    let RecGltfCtx {
        buffers,
        path,
        world,
        index_buffers,
        vertex_buffers,
        material_buffer,
    } = ctx;

    let mut meshes = world.write_storage::<Mesh>();
    let mut materials = world.write_storage::<Material>();
    let mut gltf_models = world.write_storage::<GltfModel>();
    let entities = world.read_resource::<EntitiesRes>();

    let gpu_vert_buffers: Vec<BufferHandle<VertexBuffer>> = vertex_buffers
        .into_iter()
        .map(|vert_buf| {
            renderer
                .create_resource_blocking(OwningVertexBufferDescriptor::from_raw(
                    vert_buf.data,
                    vert_buf.format.clone(),
                    BufferMutability::Immutable,
                ))
                .expect("Failed to create vertex buffer")
        })
        .collect();

    let gpu_index_buffers: Vec<BufferHandle<IndexBuffer>> = index_buffers
        .into_iter()
        .map(|idx_buf| {
            renderer
                .create_resource_blocking(OwningIndexBufferDescriptor::from_vec(
                    idx_buf,
                    BufferMutability::Immutable,
                ))
                .expect("Failed to create index buffer")
        })
        .collect();

    let gpu_uniform_buffer_handles = renderer
        .create_resource_blocking(OwningUniformBufferDescriptor::from_vec(
            material_buffer,
            BufferMutability::Immutable,
        ))
        .expect("Failed to create uniform buffer for materials")
        .split();

    for (ent, model) in (&entities, &gltf_models).join() {
        let gltf_vh = &model.vertex;
        let gltf_ih = &model.index;
        let gltf_mat = &model.mat;
        let gpu_vh = gpu_vert_buffers[gltf_vh.buf_idx];
        let gpu_ih = gpu_index_buffers[gltf_ih.buf_idx];

        let vertex_buffer = gltf_vh.as_gpu_handle(gpu_vh);
        let index_buffer = gltf_ih.as_gpu_handle(gpu_ih);
        meshes
            .insert(
                ent,
                Mesh(trekanten::mesh::Mesh {
                    vertex_buffer,
                    index_buffer,
                }),
            )
            .expect("Failed to insert mesh");

        let normal_map = gltf_mat.normal_map.as_ref().map(|x| {
            let tex_h = renderer
                .create_resource_blocking(TextureDescriptor::file(
                    x.tex.path.clone(),
                    x.tex.format,
                    MipMaps::Generate,
                ))
                .expect("Failed to create texture handle for normal map");

            let tex_use = crate::render::material::TextureUse {
                handle: tex_h,
                coord_set: 0,
            };

            crate::render::material::NormalMap {
                tex: tex_use,
                scale: x.scale,
            }
        });

        let base_color_texture = gltf_mat.base_color.as_ref().map(|t| {
            let tex_h = renderer
                .create_resource_blocking(trekanten::texture::TextureDescriptor::file(
                    t.path.clone(),
                    t.format,
                    MipMaps::Generate,
                ))
                .expect("Failed to create texture handle for normal map");
            crate::render::material::TextureUse {
                handle: tex_h,
                coord_set: 0,
            }
        });

        let metallic_roughness_texture = gltf_mat.metallic_roughness.as_ref().map(|t| {
            let tex_h = renderer
                .create_resource_blocking(trekanten::texture::TextureDescriptor::file(
                    t.path.clone(),
                    t.format,
                    MipMaps::Generate,
                ))
                .expect("Failed to create texture handle for normal map");
            crate::render::material::TextureUse {
                handle: tex_h,
                coord_set: 0,
            }
        });
        let material_uniforms = gpu_uniform_buffer_handles[gltf_mat.material.buf_idx];
        let mat_data = crate::render::material::Material::PBR {
            material_uniforms,
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
            has_vertex_colors: gltf_mat.has_vertex_colors,
        };

        materials
            .insert(ent, mat_data)
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
    let start = std::time::Instant::now();
    let (gltf_doc, buffers, _images) = gltf::import(path).expect("Unable to import gltf");
    log::trace!(
        "gltf import took {} ms",
        start.elapsed().as_secs_f32() * 1000.0
    );
    assert_eq!(gltf_doc.scenes().len(), 1);
    let mut rec_ctx = RecGltfCtx {
        buffers,
        path: path.into(),
        world,
        index_buffers: Vec::new(),
        vertex_buffers: Vec::new(),
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

        let AssetGraphResult { node, camera } = load_node_rec(&mut rec_ctx, &node);
        roots.push(node);
        camera_ent = camera;
    }

    let cam_transform = get_cam_transform(gltf_doc, rec_ctx.world, camera_ent);

    upload_to_gpu(renderer, rec_ctx);
    log::trace!("gltf asset done");

    LoadedAsset {
        scene_roots: roots,
        camera: cam_transform,
    }
}

pub fn register_components(world: &mut World) {
    world.register::<GltfModel>();
}
