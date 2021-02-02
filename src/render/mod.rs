use thiserror::Error;

use crate::ecs::prelude::*;

use trekanten::descriptor::DescriptorSet;
use trekanten::mem::{BufferMutability, OwningUniformBufferDescriptor, UniformBuffer};
use trekanten::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError, ShaderDescriptor,
};
use trekanten::resource::Handle;
use trekanten::resource::ResourceManager;
use trekanten::util;
use trekanten::vertex::VertexFormat;
use trekanten::BufferHandle;
use trekanten::RenderPassEncoder;
use trekanten::Renderer;

mod bounding_box;
pub mod debug_window;
pub mod geometry;
pub mod light;
pub mod material;
pub mod mesh;
pub mod pipeline;
pub mod ui;
pub mod uniform;

pub use mesh::GpuMesh;
use mesh::PendingMesh;

use crate::camera::*;
use crate::ecs;
use crate::math::{Mat4, ModelMatrix, Transform, Vec3};
use material::{GpuMaterial, PendingMaterial};
use ramneryd_derive::Inspect;

use debug_window::{RenderMode, RenderSettings};

pub fn camera_pos(world: &World) -> Vec3 {
    let camera_entity = ecs::get_singleton_entity::<Camera>(world);
    let transforms = world.read_storage::<Transform>();
    transforms
        .get(camera_entity)
        .expect("Could not get position component for camera")
        .position
}

pub struct FrameData {
    pub light_buffer: BufferHandle<UniformBuffer>,
    pub frame_set: Handle<DescriptorSet>,
    pub transforms_buffer: BufferHandle<UniformBuffer>,
    pub dummy_pipeline: Handle<GraphicsPipeline>,
}

fn get_view_data(world: &World) -> (Mat4, Vec3) {
    let camera_entity = ecs::get_singleton_entity::<Camera>(world);
    let transforms = world.read_storage::<Transform>();
    let rots = world.read_storage::<CameraRotationState>();

    let cam_pos = transforms
        .get(camera_entity)
        .expect("Could not get position component for camera")
        .position;

    let cam_rotation_state = rots
        .get(camera_entity)
        .expect("Could not get rotation state for camera");

    // TODO: Camera system should write to ViewMatrixResource at the end of system
    // and we should read it here.
    let view = FreeFlyCameraController::get_view_matrix_from(cam_pos, cam_rotation_state);
    log::trace!("View matrix: {:#?}", view);

    (view, cam_pos)
}

fn get_proj_matrix(aspect_ratio: f32) -> Mat4 {
    let mut proj =
        Mat4::perspective_rh_zo(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.05, 1000000.0);

    // glm::perspective is based on opengl left-handed coordinate system,
    // vulkan has the y-axis
    // inverted (right-handed upside-down).
    proj[(1, 1)] *= -1.0;

    proj
}

#[derive(Component, Default)]
#[component(storage = "NullStorage")]
pub struct ReloadMaterial;

#[derive(Component)]
#[component(inspect)]
pub struct RenderableMaterial {
    gfx_pipeline: Handle<GraphicsPipeline>,
    material_descriptor_set: Handle<DescriptorSet>,
    mode: RenderMode,
}

// TODO: Bindings here need to match with shader
fn create_material_descriptor_set(
    renderer: &mut Renderer,
    material: &GpuMaterial,
) -> Handle<DescriptorSet> {
    match &material {
        material::GpuMaterial::PBR {
            material_uniforms,
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
            ..
        } => {
            let mut desc_set_builder = DescriptorSet::builder(renderer);

            desc_set_builder = desc_set_builder.add_buffer(
                &material_uniforms,
                0,
                trekanten::pipeline::ShaderStage::FRAGMENT,
            );

            if let Some(bct) = &base_color_texture {
                desc_set_builder = desc_set_builder.add_texture(
                    &bct.handle,
                    1,
                    trekanten::pipeline::ShaderStage::FRAGMENT,
                );
            }

            if let Some(mrt) = &metallic_roughness_texture {
                desc_set_builder = desc_set_builder.add_texture(
                    &mrt.handle,
                    2,
                    trekanten::pipeline::ShaderStage::FRAGMENT,
                );
            }

            if let Some(nm) = &normal_map {
                desc_set_builder = desc_set_builder.add_texture(
                    &nm.handle,
                    3,
                    trekanten::pipeline::ShaderStage::FRAGMENT,
                );
            }

            desc_set_builder.build()
        }
        material::GpuMaterial::Unlit { color_uniform } => {
            let mut desc_set_builder = DescriptorSet::builder(renderer);
            desc_set_builder = desc_set_builder.add_buffer(
                &color_uniform,
                0,
                trekanten::pipeline::ShaderStage::FRAGMENT,
            );

            desc_set_builder.build()
        }
    }
}

#[derive(Debug, Error)]
pub enum MaterialError {
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("GLSL compiler error: {0}")]
    GlslCompiler(#[from] pipeline::CompilerError),
}

pub fn get_pipeline_for(
    renderer: &mut Renderer,
    world: &World,
    mesh: &GpuMesh,
    mat: &material::GpuMaterial,
    global_render_mode: RenderMode,
) -> Result<Handle<GraphicsPipeline>, MaterialError> {
    // TODO: Infer from spirv?
    let vertex_format = renderer
        .get_resource(&mesh.vertex_buffer)
        .expect("Invalid handle")
        .format()
        .clone();

    let polygon_mode = match (global_render_mode, mesh.polygon_mode) {
        (RenderMode::Opaque, mesh_poly_mode) => mesh_poly_mode,
        (RenderMode::Wireframe, _) => trekanten::pipeline::PolygonMode::Line,
    };

    let shader_compiler = world.read_resource::<pipeline::ShaderCompiler>();
    let pipe = match mat {
        material::GpuMaterial::PBR {
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
            has_vertex_colors,
            ..
        } => {
            // TODO: Normal map does not infer tangents at all times
            let has_nm = normal_map.is_some();
            let has_bc = base_color_texture.is_some();
            let has_mr = metallic_roughness_texture.is_some();
            let def = pipeline::pbr_gltf::ShaderDefinition {
                has_tex_coords: has_nm || has_bc || has_mr,
                has_vertex_colors: *has_vertex_colors,
                has_tangents: has_nm,
                has_base_color_texture: has_bc,
                has_metallic_roughness_texture: has_mr,
                has_normal_map: has_nm,
            };

            let (vert, frag) = pipeline::pbr_gltf::compile(&*shader_compiler, &def)?;
            let desc = GraphicsPipelineDescriptor::builder()
                .vert(ShaderDescriptor::FromRawSpirv(vert.data()))
                .frag(ShaderDescriptor::FromRawSpirv(frag.data()))
                .vertex_format(vertex_format)
                .polygon_mode(polygon_mode)
                .build()?;

            renderer.create_resource_blocking(desc)?
        }
        material::GpuMaterial::Unlit { .. } => {
            use std::path::PathBuf;
            let vertex = shader_compiler.compile(
                &pipeline::Defines::empty(),
                &PathBuf::from("pos_only_vert.glsl"),
                pipeline::ShaderType::Vertex,
            )?;
            let fragment = shader_compiler.compile(
                &pipeline::Defines::empty(),
                &PathBuf::from("uniform_color_frag.glsl"),
                pipeline::ShaderType::Fragment,
            )?;

            let desc = GraphicsPipelineDescriptor::builder()
                .vert(ShaderDescriptor::FromRawSpirv(vertex.data()))
                .frag(ShaderDescriptor::FromRawSpirv(fragment.data()))
                .vertex_format(vertex_format)
                .polygon_mode(polygon_mode)
                .build()?;

            renderer.create_resource_blocking(desc)?
        }
    };

    Ok(pipe)
}

fn create_renderable(
    renderer: &mut Renderer,
    world: &World,
    mesh: &GpuMesh,
    material: &GpuMaterial,
    render_mode: RenderMode,
) -> RenderableMaterial {
    log::trace!("Creating renderable: {:?}, {:?}", material, render_mode);
    let material_descriptor_set = create_material_descriptor_set(renderer, material);
    let gfx_pipeline = get_pipeline_for(renderer, world, mesh, &material, render_mode)
        .expect("Failed to get pipeline");
    RenderableMaterial {
        gfx_pipeline,
        material_descriptor_set,
        mode: render_mode,
    }
}

#[profiling::function]
fn create_renderables(renderer: &mut Renderer, world: &mut World, render_mode: RenderMode) {
    use specs::storage::StorageEntry;

    let meshes = world.read_storage::<GpuMesh>();
    let materials = world.read_storage::<GpuMaterial>();
    let mut should_reload = world.write_storage::<ReloadMaterial>();
    let mut renderables = world.write_storage::<RenderableMaterial>();
    let entities = world.entities();

    for (ent, mesh, mat) in (&entities, &meshes, &materials).join() {
        // TODO: Move to function
        let entry = renderables.entry(ent).expect("Failed to get entry!");
        match entry {
            StorageEntry::Occupied(mut entry) => {
                if entry.get().mode != render_mode {
                    log::trace!("Renderable did not match render mode, creating new");
                    todo!("No support for render modes yet!")
                /*
                let rend = create_renderable(&renderer, world, mat, render_mode);
                occ_entry.insert(rend)
                */
                } else {
                    log::trace!("Using existing Renderable");
                    if should_reload.contains(ent) {
                        log::trace!("Reloading shader for {:?}", ent);
                        // TODO: Destroy the previous pipeline
                        match get_pipeline_for(renderer, world, mesh, mat, render_mode) {
                            Ok(pipeline) => entry.get_mut().gfx_pipeline = pipeline,
                            Err(e) => log::error!("Failed to compile pipeline: {}", e),
                        }
                    }
                }
            }
            StorageEntry::Vacant(entry) => {
                log::trace!("No Renderable found, creating new");
                let rend = create_renderable(renderer, world, mesh, mat, render_mode);
                entry.insert(rend);
            }
        }
    }

    should_reload.clear();
}

#[profiling::function]
fn draw_entities<'a>(world: &World, cmd_buf: &mut RenderPassEncoder<'a>) {
    let model_matrices = world.read_storage::<ModelMatrix>();
    let meshes = world.read_storage::<GpuMesh>();
    let renderables = world.read_storage::<RenderableMaterial>();

    for (mesh, renderable, mtx) in (&meshes, &renderables, &model_matrices).join() {
        let trn = uniform::Model {
            model: (*mtx).into(),
            model_it: mtx.0.inverted().transposed().into_col_arrays(),
        };

        cmd_buf
            .bind_graphics_pipeline(&renderable.gfx_pipeline)
            .bind_shader_resource_group(
                1,
                &renderable.material_descriptor_set,
                &renderable.gfx_pipeline,
            )
            .bind_push_constant(
                &renderable.gfx_pipeline,
                trekanten::pipeline::ShaderStage::VERTEX,
                &trn,
            )
            .draw_mesh(&mesh.vertex_buffer, &mesh.index_buffer);
    }
}

#[profiling::function]
pub fn draw_frame(world: &mut World, ui: &mut ui::UIContext, renderer: &mut Renderer) {
    let cam_entity = ecs::try_get_singleton_entity::<Camera>(world);
    if cam_entity.is_none() {
        log::warn!("Did not find a camera entity, can't render");
        return;
    }

    let render_mode = {
        let render_settings = world.read_resource::<RenderSettings>();
        render_settings.render_mode
    };

    GpuUpload::resolve_pending(world, renderer);
    create_renderables(renderer, world, render_mode);

    let aspect_ratio = renderer.aspect_ratio();
    let mut frame = match renderer.next_frame() {
        frame @ Ok(_) => frame,
        Err(trekanten::RenderError::NeedsResize(reason)) => {
            log::debug!("Resize reason: {:?}", reason);
            renderer
                .resize(world.read_resource::<crate::io::MainWindow>().extents())
                .expect("Failed to resize renderer");
            renderer.next_frame()
        }
        e => e,
    }
    .expect("Failed to get next frame");

    let ui_draw_commands = ui.build_ui(world, &mut frame);

    let FrameData {
        light_buffer,
        frame_set,
        transforms_buffer,
        dummy_pipeline,
    } = &*world.read_resource::<FrameData>();

    // View data
    {
        let (view_matrix, view_pos) = get_view_data(world);
        let view_proj = get_proj_matrix(aspect_ratio) * view_matrix;
        let transforms = uniform::ViewData {
            view_proj: view_proj.into_col_arrays(),
            view_pos: [view_pos.x, view_pos.y, view_pos.z, 1.0f32],
        };

        // TODO: Rename transforms
        frame
            .update_uniform_blocking(transforms_buffer, &transforms)
            .expect("Failed to update uniform");
    }

    let data = light::build_light_data_uniform(world);
    frame
        .update_uniform_blocking(light_buffer, &data)
        .expect("Failed to update light");

    // main render pass
    {
        let mut builder = frame
            .begin_render_pass()
            .expect("Failed to begin render pass");

        builder
            .bind_graphics_pipeline(dummy_pipeline)
            .bind_shader_resource_group(0u32, frame_set, dummy_pipeline);

        draw_entities(world, &mut builder);
        if let Some(ui_draw_commands) = ui_draw_commands {
            ui_draw_commands.record_draw_commands(&mut builder);
        }

        let buf = builder
            .build()
            .expect("Failed to create render pass command buffer");
        frame.add_raw_command_buffer(buf);
    }

    let frame = frame.finish();
    renderer
        .submit(frame)
        .or_else(|e| {
            if let trekanten::RenderError::NeedsResize(reason) = e {
                log::info!("Resize reason: {:?}", reason);
                renderer.resize(world.read_resource::<crate::io::MainWindow>().extents())
            } else {
                Err(e)
            }
        })
        .expect("Failed to submit frame");
}

pub fn setup_resources(world: &mut World, mut renderer: &mut Renderer) {
    use uniform::UniformBlock as _;

    {
        let shader_compiler =
            pipeline::ShaderCompiler::new().expect("Failed to create shader compiler");

        world.insert(shader_compiler);
        world.insert(renderer.loader().unwrap());
    }

    let frame_data = {
        let shader_compiler = world.read_resource::<pipeline::ShaderCompiler>();

        log::trace!("Creating dummy pipeline");

        // TODO: Single elem uniform buffer here. Add to the same buffer?
        let light_data = vec![uniform::LightingData {
            punctual_lights: [uniform::PackedLight::default(); uniform::MAX_NUM_LIGHTS],
            num_lights: 0,
        }];
        let light_data =
            OwningUniformBufferDescriptor::from_vec(light_data, BufferMutability::Mutable);
        let light_data = renderer.create_resource_blocking(light_data).expect("FAIL");

        let view_data = vec![uniform::ViewData {
            view_proj: [[0.0; 4]; 4],
            view_pos: [0.0; 4],
        }];
        let view_data =
            OwningUniformBufferDescriptor::from_vec(view_data, BufferMutability::Mutable);
        let view_data = renderer.create_resource_blocking(view_data).expect("FAIL");

        assert_eq!(uniform::LightingData::SET, uniform::ViewData::SET);
        let frame_set = DescriptorSet::builder(&mut renderer)
            .add_buffer(
                &view_data,
                uniform::ViewData::BINDING,
                trekanten::pipeline::ShaderStage::VERTEX
                    | trekanten::pipeline::ShaderStage::FRAGMENT,
            )
            .add_buffer(
                &light_data,
                uniform::LightingData::BINDING,
                trekanten::pipeline::ShaderStage::FRAGMENT,
            )
            .build();

        let vertex_format = VertexFormat::builder()
            .add_attribute(util::Format::FLOAT3)
            .add_attribute(util::Format::FLOAT3)
            .build();

        let result = pipeline::pbr_gltf::compile_default(&shader_compiler);

        if let Err(e) = result {
            log::error!("{}", e);
            return;
        }

        let (vert, frag) = result.unwrap();
        let desc = GraphicsPipelineDescriptor::builder()
            .vert(ShaderDescriptor::FromRawSpirv(vert.data()))
            .frag(ShaderDescriptor::FromRawSpirv(frag.data()))
            .vertex_format(vertex_format)
            .build()
            .expect("Failed to build graphics pipeline descriptor");

        let dummy_pipeline = renderer.create_resource_blocking(desc).expect("FAIL");
        FrameData {
            light_buffer: light_data,
            frame_set,
            transforms_buffer: view_data,
            dummy_pipeline,
        }
    };

    world.insert(frame_data);
    log::trace!("Done");
}

#[derive(Debug, Clone, Inspect)]
pub enum Pending<T1, T2> {
    Pending(T1),
    Available(T2),
}

#[derive(Component)]
struct UploadAsync;
struct GpuUpload;
impl GpuUpload {
    pub const ID: &'static str = "GpuUpload";
}
impl GpuUpload {
    #[profiling::function]
    fn resolve_pending(world: &mut World, renderer: &mut Renderer) {
        use trekanten::loader::HandleMapping;
        let mut loader = world.write_resource::<trekanten::Loader>();
        let mut pending_materials = world.write_storage::<PendingMaterial>();
        let mut pending_meshes = world.write_storage::<PendingMesh>();
        let mut materials = world.write_storage::<GpuMaterial>();
        let mut meshes = world.write_storage::<GpuMesh>();
        let mut transfer_guard = loader.transfer(renderer);
        for mapping in transfer_guard.iter() {
            match mapping {
                // TODO: drain_filter()
                HandleMapping::UniformBuffer { old, new } => {
                    for (ent, _) in (&world.entities(), &pending_materials.mask().clone()).join() {
                        let pending = if let Some(pending) = pending_materials.get_mut(ent) {
                            pending
                        } else {
                            log::warn!(
                                "Uniform is done for entity {:?} but it is no longer pending",
                                ent
                            );
                            continue;
                        };

                        match pending {
                            PendingMaterial::Unlit { color_uniform } => {
                                assert!(
                                    std::matches!(color_uniform, Pending::Pending(prev) if *prev == old)
                                );
                                *color_uniform = Pending::Available(new);
                            }
                            PendingMaterial::PBR {
                                material_uniforms, ..
                            } => {
                                assert!(
                                    std::matches!(material_uniforms, Pending::Pending(prev) if *prev == old)
                                );
                                *material_uniforms = Pending::Available(new);
                            }
                        };

                        if !pending.is_done() {
                            continue;
                        }

                        let material = pending_materials
                            .remove(ent)
                            .expect("This is alive")
                            .finish();
                        materials.insert(ent, material).expect("This is alive");
                    }
                }
                HandleMapping::VertexBuffer { old, new } => {
                    for (ent, _) in (&world.entities(), &pending_meshes.mask().clone()).join() {
                        let pending = if let Some(pending) = pending_meshes.get_mut(ent) {
                            pending
                        } else {
                            log::warn!(
                                "Vertex buffer is done for entity {:?} but it is no longer pending",
                                ent
                            );
                            continue;
                        };
                        assert!(
                            std::matches!(pending.vertex_buffer, Pending::Pending(prev) if prev == old)
                        );
                        pending.vertex_buffer = Pending::Available(new);

                        if let Some(mesh) = pending.try_finish() {
                            meshes.insert(ent, mesh).expect("I'm alive!");
                            pending_meshes.remove(ent).expect("I'm alive!");
                        }
                    }
                }
                HandleMapping::IndexBuffer { old, new } => {
                    for (ent, _) in (&world.entities(), &pending_meshes.mask().clone()).join() {
                        let pending = if let Some(pending) = pending_meshes.get_mut(ent) {
                            pending
                        } else {
                            log::warn!(
                                "Index buffer is done for entity {:?} but it is no longer pending",
                                ent
                            );
                            continue;
                        };
                        assert!(
                            std::matches!(pending.index_buffer, Pending::Pending(prev) if prev == old)
                        );
                        pending.index_buffer = Pending::Available(new);

                        if let Some(mesh) = pending.try_finish() {
                            meshes.insert(ent, mesh).expect("I'm alive!");
                            pending_meshes.remove(ent).expect("I'm alive!");
                        }
                    }
                }
                HandleMapping::Texture { old, new } => {
                    for (ent, _) in (&world.entities(), &pending_materials.mask().clone()).join() {
                        let pending = if let Some(pending) = pending_materials.get_mut(ent) {
                            pending
                        } else {
                            log::warn!(
                                "Uniform is done for entity {:?} but it is no longer pending",
                                ent
                            );
                            continue;
                        };

                        match pending {
                            PendingMaterial::PBR {
                                normal_map,
                                base_color_texture,
                                metallic_roughness_texture,
                                ..
                            } => {
                                let mut found = false;
                                for tex in
                                    [normal_map, base_color_texture, metallic_roughness_texture]
                                        .iter_mut()
                                {
                                    if let Some(Pending::Pending(tex_inner)) = tex {
                                        if tex_inner.handle == old {
                                            found = true;
                                            **tex =
                                                Some(Pending::Available(material::TextureUse {
                                                    handle: new,
                                                    coord_set: tex_inner.coord_set,
                                                }));
                                            break;
                                        }
                                    }
                                }
                                assert!(found);
                            }
                            PendingMaterial::Unlit { .. } => {
                                unreachable!("Can't have pending textures for this variant")
                            }
                        };

                        if !pending.is_done() {
                            continue;
                        }

                        let material = pending_materials
                            .remove(ent)
                            .expect("This is alive")
                            .finish();
                        materials.insert(ent, material).expect("This is alive");
                    }
                }
            }
        }
    }
}

impl<'a> System<'a> for GpuUpload {
    type SystemData = (
        WriteExpect<'a, trekanten::Loader>,
        WriteStorage<'a, material::Unlit>,
        WriteStorage<'a, PendingMaterial>,
        WriteStorage<'a, mesh::CpuMesh>,
        WriteStorage<'a, PendingMesh>,
        WriteStorage<'a, UploadAsync>,
        Entities<'a>,
    );

    fn run(&mut self, data: Self::SystemData) {
        use trekanten::loader::ResourceLoader;

        let (
            loader,
            unlit_materials,
            mut pending_mats,
            cpu_meshes,
            mut pending_meshes,
            mut command_markers,
            entities,
        ) = data;

        let mut ubuf = Vec::new();
        for (ent, unlit) in (&entities, &unlit_materials).join() {
            if let StorageEntry::Vacant(_) = pending_mats.entry(ent).unwrap() {
                ubuf.push(uniform::UnlitUniformData {
                    color: unlit.color.into_array(),
                });
            }
        }

        if !ubuf.is_empty() {
            let async_handle = loader
                .load(OwningUniformBufferDescriptor::from_vec(
                    ubuf,
                    BufferMutability::Immutable,
                ))
                .expect("Failed to load uniform buffer");
            for (i, (ent, _unlit)) in (&entities, &unlit_materials).join().enumerate() {
                if let StorageEntry::Vacant(entry) = pending_mats.entry(ent).unwrap() {
                    entry.insert(PendingMaterial::Unlit {
                        color_uniform: Pending::Pending(BufferHandle::sub_buffer(
                            async_handle,
                            i as u32,
                            1,
                        )),
                    });
                }
            }
        }

        for (ent, mesh, _) in (&entities, &cpu_meshes, &command_markers).join() {
            pending_meshes
                .insert(ent, PendingMesh::load(&loader, &mesh))
                .expect("I'm alive");
        }

        command_markers.clear();
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    [
        debug_window::register_systems,
        bounding_box::register_systems,
        light::register_systems,
    ]
    .iter()
    .fold(builder, |a, x| x(a))
    .with(GpuUpload, GpuUpload::ID, &[])
}
