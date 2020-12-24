use crate::ecs;
use ecs::prelude::*;
use std::path::{Path, PathBuf};

use trekanten::loader::ResourceLoader;
use trekanten::mesh::BufferMutability;
use trekanten::mesh::{
    IndexBuffer, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor, VertexBuffer,
};
use trekanten::texture::{MipMaps, Texture, TextureDescriptor};
use trekanten::uniform::{OwningUniformBufferDescriptor, UniformBuffer};
use trekanten::util;
use trekanten::vertex::VertexFormat;
use trekanten::BufferHandle;
use trekanten::Handle;

use crate::camera::Camera;
use crate::common::Name;
use crate::graph;
use crate::math::*;
use crate::render;
use crate::render::material::{Material, TextureUse};
use crate::render::uniform::PBRMaterialData;

fn load_texture(
    ctx: &RecGltfCtx,
    texture: &gltf::texture::Texture,
    coord_set: u32,
    format: util::Format,
) -> Handle<Texture> {
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

    ctx.data.loader.load(TextureDescriptor::file(
        image_path,
        format,
        MipMaps::Generate,
    ))
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
) -> (OwningVertexBufferDescriptor, bool) {
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

    (
        OwningVertexBufferDescriptor::from_raw(data, format, BufferMutability::Immutable),
        has_vertex_colors,
    )
}

fn to_index_buffer(indices: gltf::mesh::util::ReadIndices<'_>) -> OwningIndexBufferDescriptor {
    use gltf::mesh::util::ReadIndices;
    match indices {
        ReadIndices::U8(iter) => {
            let v: Vec<u16> = iter.map(|byte| byte as u16).collect();
            OwningIndexBufferDescriptor::from_vec(v, BufferMutability::Immutable)
        }
        ReadIndices::U16(iter) => {
            let v: Vec<u16> = iter.collect();
            OwningIndexBufferDescriptor::from_vec(v, BufferMutability::Immutable)
        }
        ReadIndices::U32(iter) => {
            let v: Vec<u32> = iter.collect();
            OwningIndexBufferDescriptor::from_vec(v, BufferMutability::Immutable)
        }
    }
}

fn load_primitive<'a>(ctx: &mut RecGltfCtx, primitive: &gltf::Primitive<'a>) -> PendingGltfModel {
    assert!(primitive.mode() == gltf::mesh::Mode::Triangles);
    let reader = primitive.reader(|buffer| Some(&ctx.buffers[buffer.index()]));

    let triangle_index_data = reader.read_indices().expect("Found no indices");

    let index = ctx.data.loader.load(to_index_buffer(triangle_index_data));
    let (vertex, has_vertex_colors) = interleave_vertex_buffer(ctx, primitive);
    let vertex = ctx.data.loader.load(vertex);

    let mat = primitive.material();
    let pbr_mr = mat.pbr_metallic_roughness();
    if mat.emissive_texture().is_some() {
        unimplemented!("No support for emissive texture!");
    }

    let base_color_texture = pbr_mr.base_color_texture().map(|info| {
        load_texture(
            ctx,
            &info.texture(),
            info.tex_coord(),
            util::Format::RGBA_SRGB,
        )
    });

    let metallic_roughness_texture = pbr_mr.metallic_roughness_texture().map(|info| {
        load_texture(
            ctx,
            &info.texture(),
            info.tex_coord(),
            util::Format::RGBA_UNORM,
        )
    });

    let normal_map = mat.normal_texture().map(|normal_map| {
        load_texture(
            ctx,
            &normal_map.texture(),
            normal_map.tex_coord(),
            util::Format::RGBA_UNORM,
        )
    });

    let material = PBRMaterialData {
        base_color_factor: pbr_mr.base_color_factor(),
        metallic_factor: pbr_mr.metallic_factor(),
        roughness_factor: pbr_mr.roughness_factor(),
        normal_scale: mat.normal_texture().map(|nm| nm.scale()).unwrap_or(1.0),
        _padding: 0.0,
    };
    let idx = ctx.material_buffer.len();
    ctx.material_buffer.push(material);

    PendingGltfModel {
        mat: GltfMaterial {
            material_data: UniformHandle::CPU(idx),
            normal_map,
            base_color: base_color_texture,
            metallic_roughness: metallic_roughness_texture,
            has_vertex_colors,
        },
        vertex,
        index,
    }
}

fn get_transform(src: gltf::scene::Transform) -> Transform {
    let (pos, rot, scale) = src.decomposed();
    let mut t = Transform::identity();
    t.position = Vec3::from(pos);
    t.rotation = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]).normalized();
    if !scale.iter().all(|x| (*x - scale[0]).abs() < f32::EPSILON) {
        log::warn!("Non-uniform scaling in asset: {:?}", scale);
        log::warn!("Using only {}", scale[0]);
    }
    t.scale = scale[0];

    t
}

fn load_node_rec(ctx: &mut RecGltfCtx, src: &gltf::Node) -> ecs::Entity {
    let tfm = get_transform(src.transform());

    let mut node = ctx
        .data
        .entities
        .build_entity()
        .with(tfm, &mut ctx.data.transforms);

    if let Some(name) = src.name() {
        node = node.with(Name::from(name), &mut ctx.data.names);
    }

    let node = node.build();

    if let Some(mesh) = src.mesh() {
        let mesh_child = ctx.data.entities.create();
        for primitive in mesh.primitives() {
            let gltf_model = load_primitive(ctx, &primitive);
            let prim_child = ctx
                .data
                .entities
                .build_entity()
                .with(gltf_model, ctx.data.gltf_models)
                .build();
            graph::add_edge_sys(
                &mut ctx.data.children_storage,
                &mut ctx.data.parent_storage,
                mesh_child,
                prim_child,
            );
        }
        graph::add_edge_sys(
            &mut ctx.data.children_storage,
            &mut ctx.data.parent_storage,
            node,
            mesh_child,
        );

        if let Some(name) = mesh.name() {
            ctx.data
                .names
                .insert(mesh_child, Name::from(name))
                .expect("Failed to insert name");
        }
    }

    /*
     * TODO: Handle cameras
    if src.camera().is_some() {
        ctx.data
            .cameras
            .insert(node, Camera {})
            .expect("Failed to insert camera marker");
    }
    */

    for gltf_child in src.children() {
        let child = load_node_rec(ctx, &gltf_child);
        graph::add_edge_sys(
            &mut ctx.data.children_storage,
            &mut ctx.data.parent_storage,
            node,
            child,
        );
    }

    node
}

fn get_cam_transform(
    gltf_doc: gltf::Document,
    world: &World,
    camera_ent: Option<ecs::Entity>,
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

pub fn load_asset(world: &mut World, path: &Path) {
    world
        .create_entity()
        .with(GltfAsset {
            path: PathBuf::from(path),
        })
        .build();
}

#[derive(Default, Component)]
pub struct GltfAsset {
    path: PathBuf,
}

#[derive(Debug)]
enum UniformHandle {
    CPU(usize),
    GPU(BufferHandle<UniformBuffer>),
}

impl UniformHandle {
    fn gpu_handle(&self) -> BufferHandle<UniformBuffer> {
        match self {
            UniformHandle::GPU(h) => h.clone(),
            _ => panic!("Expected a gpu handle"),
        }
    }
}

#[derive(Debug)]
struct GltfMaterial {
    material_data: UniformHandle,
    normal_map: Option<Handle<Texture>>,
    base_color: Option<Handle<Texture>>,
    metallic_roughness: Option<Handle<Texture>>,
    has_vertex_colors: bool,
}

#[derive(Component)]
#[component(storage = "VecStorage")]
pub struct PendingGltfModel {
    mat: GltfMaterial,
    index: BufferHandle<IndexBuffer>,
    vertex: BufferHandle<VertexBuffer>,
}

impl PendingGltfModel {
    fn is_done(&self, loader: &trekanten::Loader) -> bool {
        let tex_done = |tex: &Option<Handle<Texture>>| -> bool {
            tex.map(|t| loader.is_done(&t).unwrap()).unwrap_or(true)
        };

        loader.is_done(&self.vertex).unwrap()
            && loader.is_done(&self.index).unwrap()
            && tex_done(&self.mat.normal_map)
            && tex_done(&self.mat.base_color)
            && tex_done(&self.mat.metallic_roughness)
            && loader
                .is_done(&self.mat.material_data.gpu_handle())
                .unwrap()
    }
}

struct GltfLoader;

impl GltfLoader {
    pub const ID: &'static str = "GltfLoader";
}

#[derive(SystemData)]
struct LoaderData<'a> {
    entities: Entities<'a>,
    loader: ReadExpect<'a, trekanten::Loader>,
    assets: WriteStorage<'a, GltfAsset>,
    transforms: WriteStorage<'a, Transform>,
    parent_storage: WriteStorage<'a, graph::Parent>,
    children_storage: WriteStorage<'a, graph::Children>,
    names: WriteStorage<'a, Name>,
    gltf_models: WriteStorage<'a, PendingGltfModel>,
    cameras: WriteStorage<'a, Camera>,
}

struct CtxData<'a, 'b> {
    entities: &'b mut Entities<'a>,
    transforms: &'b mut WriteStorage<'a, Transform>,
    parent_storage: &'b mut WriteStorage<'a, graph::Parent>,
    children_storage: &'b mut WriteStorage<'a, graph::Children>,
    names: &'b mut WriteStorage<'a, Name>,
    gltf_models: &'b mut WriteStorage<'a, PendingGltfModel>,
    cameras: &'b mut WriteStorage<'a, Camera>,
    loader: &'b ReadExpect<'a, trekanten::Loader>,
}

struct RecGltfCtx<'a, 'b> {
    pub data: CtxData<'a, 'b>,
    pub buffers: Vec<gltf::buffer::Data>,
    pub path: PathBuf,
    pub material_buffer: Vec<PBRMaterialData>,
}

impl<'a> System<'a> for GltfLoader {
    type SystemData = LoaderData<'a>;

    fn run(&mut self, data: Self::SystemData) {
        let Self::SystemData {
            mut entities,
            loader,
            mut assets,
            mut transforms,
            mut children_storage,
            mut parent_storage,
            mut names,
            mut gltf_models,
            mut cameras,
        } = data;

        let asset_entities: Vec<ecs::Entity> =
            (&entities, &assets).join().map(|(x, _)| x).collect();
        for ent in asset_entities.into_iter() {
            let asset = assets.get(ent).expect("Just filtered on this!");
            log::trace!("load gltf asset {}", asset.path.display());

            let start = std::time::Instant::now();
            let (gltf_doc, buffers, _images) =
                gltf::import(&asset.path).expect("Unable to import gltf");
            log::trace!(
                "gltf import took {} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );

            let ctx_data = CtxData {
                entities: &mut entities,
                transforms: &mut transforms,
                parent_storage: &mut parent_storage,
                children_storage: &mut children_storage,
                names: &mut names,
                gltf_models: &mut gltf_models,
                cameras: &mut cameras,
                loader: &loader,
            };
            assert_eq!(gltf_doc.scenes().len(), 1);
            let mut rec_ctx = RecGltfCtx {
                buffers,
                path: asset.path.clone(),
                data: ctx_data,
                material_buffer: Vec::new(),
            };

            // A scene may have several root nodes
            let nodes = gltf_doc.scenes().next().expect("No scenes!").nodes();
            if gltf_doc.scenes().len() > 1 {
                log::warn!("More than one scene found, only displaying the first");
                log::warn!("Number of scenes: {}", gltf_doc.scenes().len());
            }
            for node in nodes {
                log::trace!("Root node {}", node.name().unwrap_or("node_no_name"));
                log::trace!("# children {}", node.children().len());

                let root = load_node_rec(&mut rec_ctx, &node);
                graph::add_edge_sys(
                    &mut rec_ctx.data.children_storage,
                    &mut rec_ctx.data.parent_storage,
                    ent,
                    root,
                );
            }

            let RecGltfCtx {
                material_buffer, ..
            } = rec_ctx;

            let gpu_uniform_buffer_handles = loader
                .load(OwningUniformBufferDescriptor::from_vec(
                    material_buffer,
                    BufferMutability::Immutable,
                ))
                .split();

            let map_cpu_to_gpu = |node: ecs::Entity| {
                if let Some(model) = gltf_models.get_mut(node) {
                    let idx = match model.mat.material_data {
                        UniformHandle::CPU(i) => i,
                        _ => panic!("Expected only cpu data at this point"),
                    };
                    model.mat.material_data = UniformHandle::GPU(gpu_uniform_buffer_handles[idx]);
                }
            };

            graph::breadth_first_sys(&children_storage, ent, map_cpu_to_gpu);

            log::trace!("gltf asset done");
        }
        assets.clear();
    }
}

struct GltfFinish;

impl GltfFinish {
    pub const ID: &'static str = "GltfFinish";
}

impl<'a> System<'a> for GltfFinish {
    type SystemData = (
        Entities<'a>,
        ReadExpect<'a, trekanten::Loader>,
        WriteStorage<'a, PendingGltfModel>,
        WriteStorage<'a, render::GpuMesh>,
        WriteStorage<'a, Material>,
    );

    fn run(&mut self, (entities, loader, mut models, mut meshes, mut materials): Self::SystemData) {
        let mut done = specs::BitSet::new();

        for (ent, model) in (&entities, &mut models).join() {
            let is_done = model.is_done(&loader);
            if is_done {
                meshes
                    .insert(
                        ent,
                        render::GpuMesh(trekanten::mesh::Mesh {
                            vertex_buffer: model.vertex,
                            index_buffer: model.index,
                        }),
                    )
                    .expect("Failed to insert mesh");

                let map_tex = |t: Option<Handle<Texture>>| -> Option<TextureUse> {
                    t.map(|handle| TextureUse {
                        handle,
                        coord_set: 0,
                    })
                };

                let mat_data = Material::PBR {
                    material_uniforms: model.mat.material_data.gpu_handle(),
                    normal_map: map_tex(model.mat.normal_map),
                    base_color_texture: map_tex(model.mat.base_color),
                    metallic_roughness_texture: map_tex(model.mat.metallic_roughness),
                    has_vertex_colors: model.mat.has_vertex_colors,
                };

                materials
                    .insert(ent, mat_data)
                    .expect("Failed to insert material");

                done.add(ent.id());
            }
        }

        for (ent, _done) in (&entities, &done).join() {
            models.remove(ent);
        }
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(GltfLoader, GltfLoader::ID, &[]).with(
        GltfFinish,
        GltfFinish::ID,
        &[GltfLoader::ID],
    )
}
