use crate::ecs;
use crate::ecs::prelude::*;
use std::path::{Path, PathBuf};

use crate::render::{HostIndexBuffer, HostVertexBuffer};
use trekant::util;
use trekant::{MipMaps, TextureDescriptor};
use trekant::{VertexBufferType, VertexFormat};

use ram_derive::Visitable;

use crate::camera::Camera;
use crate::common::Name;
use crate::graph::sys as graph;
use crate::math::*;
use crate::render;
use crate::render::material::{PhysicallyBased, TextureUse2};
use crate::render::mesh::Mesh;

fn load_texture(
    ctx: &RecGltfCtx,
    texture: &gltf::texture::Texture,
    coord_set: u32,
    format: util::Format,
) -> TextureUse2 {
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

    TextureUse2 {
        coord_set,
        desc: TextureDescriptor::File {
            path: image_path,
            format,
            mipmaps: MipMaps::None,
            ty: trekant::TextureType::Tex2D,
        },
    }
}

fn check_supported(primitive: &gltf::Primitive<'_>) {
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
fn interleave_vertex_buffer(
    ctx: &RecGltfCtx,
    primitive: &gltf::Primitive<'_>,
) -> (HostVertexBuffer, bool) {
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
    // TODO: Generalize with format writer
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
        unsafe { HostVertexBuffer::from_raw(data, VertexBufferType::new(format)) },
        has_vertex_colors,
    )
}

fn to_index_buffer(indices: gltf::mesh::util::ReadIndices<'_>) -> HostIndexBuffer {
    use gltf::mesh::util::ReadIndices;
    match indices {
        ReadIndices::U8(iter) => {
            let v: Vec<u16> = iter.map(|byte| byte as u16).collect();
            HostIndexBuffer::from_vec(v)
        }
        ReadIndices::U16(iter) => {
            let v: Vec<u16> = iter.collect();
            HostIndexBuffer::from_vec(v)
        }
        ReadIndices::U32(iter) => {
            let v: Vec<u32> = iter.collect();
            HostIndexBuffer::from_vec(v)
        }
    }
}

fn load_primitive(ctx: &mut RecGltfCtx, primitive: &gltf::Primitive<'_>) -> PendingGltfModel {
    assert!(primitive.mode() == gltf::mesh::Mode::Triangles);
    let reader = primitive.reader(|buffer| Some(&ctx.buffers[buffer.index()]));

    let triangle_index_data = reader.read_indices().expect("Found no indices");
    let index_buffer = to_index_buffer(triangle_index_data);
    let (vertex_buffer, has_vertex_colors) = interleave_vertex_buffer(ctx, primitive);

    let mesh = Mesh::new(vertex_buffer, index_buffer);

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

    let material = PhysicallyBased {
        base_color_factor: Rgba::from(pbr_mr.base_color_factor()),
        metallic_factor: pbr_mr.metallic_factor(),
        roughness_factor: pbr_mr.roughness_factor(),
        normal_scale: mat.normal_texture().map(|nm| nm.scale()).unwrap_or(1.0),
        normal_map,
        base_color_texture,
        metallic_roughness_texture,
        has_vertex_colors,
    };

    PendingGltfModel { material, mesh }
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
        .with(tfm, ctx.data.transforms);

    if let Some(name) = src.name() {
        node = node.with(Name::from(name), ctx.data.names);
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
            let PendingGltfModel { mesh, material } = load_primitive(ctx, &primitive);

            let bbox = Aabb {
                min: Vec3::from(primitive.bounding_box().min),
                max: Vec3::from(primitive.bounding_box().max),
            };

            let prim_child = ctx
                .data
                .entities
                .build_entity()
                .with(Name(format!("Primitive {}", i)), ctx.data.names)
                .with(bbox, ctx.data.bboxes)
                .with(Transform::identity(), ctx.data.transforms)
                .with(mesh, ctx.data.meshes)
                .with(material, ctx.data.pb_materials)
                .build();
            graph::add_edge(
                ctx.data.children_storage,
                ctx.data.parent_storage,
                mesh_child,
                prim_child,
            );
        }
        graph::add_edge(
            ctx.data.children_storage,
            ctx.data.parent_storage,
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
        graph::add_edge(
            ctx.data.children_storage,
            ctx.data.parent_storage,
            node,
            child,
        );
    }

    node
}

#[allow(dead_code)]
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

#[derive(Default, Component, Visitable)]
pub struct GltfAsset {
    path: PathBuf,
}

#[derive(Default, Component)]
pub struct LoadGltfAsset {
    path: PathBuf,
}

#[derive(Component, Visitable)]
pub struct PendingGltfModel {
    mesh: Mesh,
    material: PhysicallyBased,
}

struct GltfLoader;

impl GltfLoader {
    pub const ID: &'static str = "GltfLoader";
}

#[derive(SystemData)]
struct LoaderData<'a> {
    entities: Entities<'a>,
    load_assets: WriteStorage<'a, LoadGltfAsset>,
    transforms: WriteStorage<'a, Transform>,
    parent_storage: WriteStorage<'a, graph::Parent>,
    children_storage: WriteStorage<'a, graph::Children>,
    names: WriteStorage<'a, Name>,
    meshes: WriteStorage<'a, render::mesh::Mesh>,
    pb_materials: WriteStorage<'a, render::material::PhysicallyBased>,
    bboxes: WriteStorage<'a, Aabb>,
    cameras: WriteStorage<'a, Camera>,
}

struct CtxData<'a, 'b> {
    entities: &'b Entities<'a>,
    transforms: &'b mut WriteStorage<'a, Transform>,
    parent_storage: &'b mut WriteStorage<'a, graph::Parent>,
    children_storage: &'b mut WriteStorage<'a, graph::Children>,
    names: &'b mut WriteStorage<'a, Name>,
    meshes: &'b mut WriteStorage<'a, Mesh>,
    pb_materials: &'b mut WriteStorage<'a, render::material::PhysicallyBased>,
    #[allow(dead_code)]
    cameras: &'b mut WriteStorage<'a, Camera>,
    bboxes: &'b mut WriteStorage<'a, Aabb>,
}

struct RecGltfCtx<'a, 'b> {
    pub data: CtxData<'a, 'b>,
    pub buffers: Vec<gltf::buffer::Data>,
    pub path: PathBuf,
}

impl<'a> System<'a> for GltfLoader {
    type SystemData = LoaderData<'a>;

    fn run(&mut self, data: Self::SystemData) {
        let Self::SystemData {
            entities,
            mut load_assets,
            mut transforms,
            mut children_storage,
            mut parent_storage,
            mut names,
            mut meshes,
            mut pb_materials,
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
                cameras: &mut cameras,
                bboxes: &mut bboxes,
                pb_materials: &mut pb_materials,
                meshes: &mut meshes,
            };
            assert_eq!(gltf_doc.scenes().len(), 1);
            let mut rec_ctx = RecGltfCtx {
                buffers,
                path: asset.path.clone(),
                data: ctx_data,
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
                graph::add_edge(
                    rec_ctx.data.children_storage,
                    rec_ctx.data.parent_storage,
                    ent,
                    root,
                );
                rec_ctx
                    .data
                    .transforms
                    .insert(ent, Transform::identity())
                    .unwrap();
            }
        }
        load_assets.clear();
    }
}

pub struct GltfModule;

impl crate::Module for GltfModule {
    fn load(&mut self, loader: &mut crate::ModuleLoader) {
        loader.add_system(GltfLoader, GltfLoader::ID, &[]);
    }
}
