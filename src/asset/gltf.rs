use crate::ecs;
use ecs::prelude::*;
use std::path::{Path, PathBuf};

use trekanten::loader::ResourceLoader;
use trekanten::mesh::BufferMutability;
use trekanten::mesh::{
    IndexBuffer, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor, VertexBuffer,
};
use trekanten::texture::{MipMaps, Texture, TextureDescriptor};
use trekanten::uniform::OwningUniformBufferDescriptor;
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
use ramneryd_derive::Inspect;

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
            material_idx: idx,
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
        let mesh_child = ctx
            .data
            .entities
            .build_entity()
            .with(Transform::identity(), ctx.data.transforms)
            .build();

        for (i, primitive) in mesh.primitives().enumerate() {
            let gltf_model = load_primitive(ctx, &primitive);

            let bbox = BoundingBox {
                min: Vec3::from(primitive.bounding_box().min),
                max: Vec3::from(primitive.bounding_box().max),
            };

            let prim_child = ctx
                .data
                .entities
                .build_entity()
                .with(gltf_model, ctx.data.gltf_models)
                .with(Name(format!("Primitive {}", i)), ctx.data.names)
                .with(bbox, ctx.data.bboxes)
                .with(Transform::identity(), ctx.data.transforms)
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
        .with(LoadGltfAsset {
            path: PathBuf::from(path),
        })
        .build();
}

#[derive(Default, Component)]
#[component(inspect)]
pub struct GltfAsset {
    path: PathBuf,
}

#[derive(Default, Component)]
pub struct LoadGltfAsset {
    path: PathBuf,
}

#[derive(Debug, Inspect)]
struct GltfMaterial {
    material_idx: usize,
    normal_map: Option<Handle<Texture>>,
    base_color: Option<Handle<Texture>>,
    metallic_roughness: Option<Handle<Texture>>,
    has_vertex_colors: bool,
}

#[derive(Component)]
#[component(storage = "VecStorage", inspect)]
pub struct PendingGltfModel {
    mat: GltfMaterial,
    index: BufferHandle<IndexBuffer>,
    vertex: BufferHandle<VertexBuffer>,
}

struct GltfLoader;

impl GltfLoader {
    pub const ID: &'static str = "GltfLoader";
}

#[derive(SystemData)]
struct LoaderData<'a> {
    entities: Entities<'a>,
    loader: WriteExpect<'a, trekanten::Loader>,
    load_assets: WriteStorage<'a, LoadGltfAsset>,
    transforms: WriteStorage<'a, Transform>,
    parent_storage: WriteStorage<'a, graph::Parent>,
    children_storage: WriteStorage<'a, graph::Children>,
    names: WriteStorage<'a, Name>,
    gltf_models: WriteStorage<'a, PendingGltfModel>,
    meshes: WriteStorage<'a, render::mesh::PendingMesh>,
    materials: WriteStorage<'a, render::material::PendingMaterial>,
    bboxes: WriteStorage<'a, BoundingBox>,
    cameras: WriteStorage<'a, Camera>,
}

struct CtxData<'a, 'b> {
    entities: &'b Entities<'a>,
    transforms: &'b mut WriteStorage<'a, Transform>,
    parent_storage: &'b mut WriteStorage<'a, graph::Parent>,
    children_storage: &'b mut WriteStorage<'a, graph::Children>,
    names: &'b mut WriteStorage<'a, Name>,
    gltf_models: &'b mut WriteStorage<'a, PendingGltfModel>,
    cameras: &'b mut WriteStorage<'a, Camera>,
    bboxes: &'b mut WriteStorage<'a, BoundingBox>,
    loader: &'b WriteExpect<'a, trekanten::Loader>,
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
            entities,
            loader,
            mut load_assets,
            mut transforms,
            mut children_storage,
            mut parent_storage,
            mut names,
            mut gltf_models,
            mut meshes,
            mut materials,
            mut cameras,
            mut bboxes,
        } = data;

        for (ent, _) in (&entities, &load_assets).join() {
            let asset = load_assets.get(ent).expect("Just filtered on this!");
            log::trace!("load gltf asset {}", asset.path.display());

            let start = std::time::Instant::now();
            let (gltf_doc, buffers, _images) =
                gltf::import(&asset.path).expect("Unable to import gltf");
            log::trace!(
                "gltf import took {} ms",
                start.elapsed().as_secs_f32() * 1000.0
            );

            let ctx_data = CtxData {
                entities: &entities,
                transforms: &mut transforms,
                parent_storage: &mut parent_storage,
                children_storage: &mut children_storage,
                names: &mut names,
                gltf_models: &mut gltf_models,
                cameras: &mut cameras,
                bboxes: &mut bboxes,
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
                rec_ctx
                    .data
                    .transforms
                    .insert(ent, Transform::identity())
                    .unwrap();
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
                    meshes
                        .insert(
                            node,
                            render::mesh::PendingMesh::new(trekanten::mesh::Mesh {
                                vertex_buffer: model.vertex.clone(),
                                index_buffer: model.index.clone(),
                            }),
                        )
                        .unwrap();

                    let map_tex = |tex: &Option<Handle<Texture>>| -> Option<TextureUse> {
                        tex.map(|t| TextureUse {
                            handle: t.clone(),
                            coord_set: 0,
                        })
                    };

                    let idx = model.mat.material_idx;
                    materials
                        .insert(
                            node,
                            render::material::PendingMaterial::from(Material::PBR {
                                material_uniforms: gpu_uniform_buffer_handles[idx],
                                normal_map: map_tex(&model.mat.normal_map),
                                base_color_texture: map_tex(&model.mat.base_color),
                                metallic_roughness_texture: map_tex(&model.mat.metallic_roughness),
                                has_vertex_colors: model.mat.has_vertex_colors,
                            }),
                        )
                        .unwrap();
                }
            };

            graph::breadth_first_sys(&children_storage, ent, map_cpu_to_gpu);
            gltf_models.clear();

            log::trace!("gltf asset done");
        }
        load_assets.clear();
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(GltfLoader, GltfLoader::ID, &[])
}
