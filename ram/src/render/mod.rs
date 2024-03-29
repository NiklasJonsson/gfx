use thiserror::Error;

use crate::ecs::prelude::*;

use trekant::buffer::{BufferMutability, DeviceUniformBuffer, UniformBufferDescriptor};
use trekant::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError, ShaderDescriptor,
};
use trekant::pipeline_resource::PipelineResourceSet;
use trekant::resource::Handle;
use trekant::resource::ResourceManager;
use trekant::util;
use trekant::vertex::VertexFormat;
use trekant::BufferHandle;
use trekant::RenderPassEncoder;
use trekant::Renderer;

pub mod debug;
pub mod geometry;
pub mod imgui;
pub mod light;
pub mod material;
pub mod mesh;
pub mod shader;
pub mod uniform;

pub use geometry::Shape;
pub use light::Light;

use mesh::Mesh;

use crate::camera::*;
use crate::ecs;
use crate::math::{Mat4, ModelMatrix, Transform, Vec3};
use material::{GpuMaterial, PendingMaterial};
use ram_derive::Visitable;

/// Utility for working with asynchronously uploaded gpu resources
#[derive(Debug, Clone, Visitable)]
pub enum GpuResource<InFlightT, AvailT> {
    None,
    InFlight(InFlightT),
    Available(AvailT),
}
type GpuBuffer<BT> = GpuResource<BufferHandle<Async<BT>>, BufferHandle<BT>>;

pub fn camera_pos(world: &World) -> Vec3 {
    let camera_entity = ecs::get_singleton_entity::<MainRenderCamera>(world);
    let transforms = world.read_storage::<Transform>();
    transforms
        .get(camera_entity)
        .expect("Could not get position component for camera")
        .position
}

struct UnlitPassResources {
    dummy_pipeline: Handle<GraphicsPipeline>,
}

struct PBRPassResources {
    dummy_pipeline: Handle<GraphicsPipeline>,
}

struct EngineShaderResources {
    view_data: BufferHandle<DeviceUniformBuffer>,
    lighting_data: BufferHandle<DeviceUniformBuffer>,
    shadow_data: BufferHandle<DeviceUniformBuffer>,
    desc_set: Handle<PipelineResourceSet>,
}

pub struct FrameResources {
    main_render_pass: Handle<trekant::RenderPass>,
    engine_shader_resources: EngineShaderResources,
    unlit_resources: UnlitPassResources,
    pbr_resources: PBRPassResources,
    shadow: light::ShadowResources,
}

#[derive(Component, Default)]
#[component(storage = "NullStorage")]
pub struct MainRenderCamera;

#[derive(Component, Default)]
#[component(storage = "NullStorage")]
pub struct ReloadMaterial;

#[derive(Component, Visitable)]
pub struct RenderableMaterial {
    gfx_pipeline: Handle<GraphicsPipeline>,
    shadow_pipeline: Option<Handle<GraphicsPipeline>>,
    material_descriptor_set: Handle<PipelineResourceSet>,
}

// TODO: Bindings here need to match with shader
fn create_material_descriptor_set(
    renderer: &mut Renderer,
    material: &GpuMaterial,
) -> Handle<PipelineResourceSet> {
    match &material {
        material::GpuMaterial::PBR {
            material_uniforms,
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
            ..
        } => {
            let mut desc_set_builder = PipelineResourceSet::builder(renderer);

            desc_set_builder = desc_set_builder.add_buffer(
                material_uniforms,
                0,
                trekant::pipeline::ShaderStage::FRAGMENT,
            );

            if let Some(bct) = &base_color_texture {
                desc_set_builder = desc_set_builder.add_texture(
                    &bct.handle,
                    1,
                    trekant::pipeline::ShaderStage::FRAGMENT,
                    false,
                );
            }

            if let Some(mrt) = &metallic_roughness_texture {
                desc_set_builder = desc_set_builder.add_texture(
                    &mrt.handle,
                    2,
                    trekant::pipeline::ShaderStage::FRAGMENT,
                    false,
                );
            }

            if let Some(nm) = &normal_map {
                desc_set_builder = desc_set_builder.add_texture(
                    &nm.handle,
                    3,
                    trekant::pipeline::ShaderStage::FRAGMENT,
                    false,
                );
            }

            desc_set_builder.build()
        }
        material::GpuMaterial::Unlit { color_uniform, .. } => {
            PipelineResourceSet::builder(renderer)
                .add_buffer(color_uniform, 0, trekant::pipeline::ShaderStage::FRAGMENT)
                .build()
        }
    }
}

#[derive(Debug, Error)]
pub enum MaterialError {
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("GLSL compiler error: {0}")]
    GlslCompiler(#[from] shader::CompilerError),
}

fn unlit_pipeline_desc(
    shader_compiler: &shader::ShaderCompiler,
    vertex_format: VertexFormat,
    polygon_mode: trekant::pipeline::PolygonMode,
) -> Result<GraphicsPipelineDescriptor, MaterialError> {
    let vert = shader_compiler.compile(
        &shader::ShaderLocation::builtin("render/shaders/unlit/vert.glsl"),
        &shader::Defines::empty(),
        shader::ShaderType::Vertex,
    )?;
    let frag = shader_compiler.compile(
        &shader::ShaderLocation::builtin("render/shaders/unlit/frag.glsl"),
        &shader::Defines::empty(),
        shader::ShaderType::Fragment,
    )?;

    let vert = ShaderDescriptor {
        debug_name: Some("unlit-vert".to_owned()),
        spirv_code: vert.data(),
    };
    let frag = ShaderDescriptor {
        debug_name: Some("unlit-frag".to_owned()),
        spirv_code: frag.data(),
    };
    Ok(GraphicsPipelineDescriptor::builder()
        .vert(vert)
        .frag(frag)
        .vertex_format(vertex_format)
        .culling(trekant::pipeline::TriangleCulling::None)
        .polygon_mode(polygon_mode)
        .build()?)
}

fn get_pipeline_for(
    renderer: &mut Renderer,
    world: &World,
    mesh: &Mesh,
    mat: &material::GpuMaterial,
    // TODO: Clean this up
    use_builtin: bool,
) -> Result<Handle<GraphicsPipeline>, MaterialError> {
    let vertex_format = mesh.cpu_vertex_buffer.format().clone();

    let frame_data = world.read_resource::<FrameResources>();
    let shader_compiler = world.read_resource::<shader::ShaderCompiler>();
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
            let def = shader::pbr_gltf::ShaderDefinition {
                has_tex_coords: has_nm || has_bc || has_mr,
                has_vertex_colors: *has_vertex_colors,
                has_tangents: has_nm,
                has_base_color_texture: has_bc,
                has_metallic_roughness_texture: has_mr,
                has_normal_map: has_nm,
            };

            let (vert, frag) = shader::pbr_gltf::compile(&*shader_compiler, &def, use_builtin)?;

            let vert = ShaderDescriptor {
                debug_name: Some("pbr-vert".to_owned()),
                spirv_code: vert.data(),
            };
            let frag = ShaderDescriptor {
                debug_name: Some("pbr-frag".to_owned()),
                spirv_code: frag.data(),
            };
            let desc = GraphicsPipelineDescriptor::builder()
                .vert(vert)
                .frag(frag)
                .vertex_format(vertex_format)
                .polygon_mode(trekant::pipeline::PolygonMode::Fill)
                .build()?;

            renderer.create_gfx_pipeline(desc, &frame_data.main_render_pass)?
        }
        material::GpuMaterial::Unlit { polygon_mode, .. } => {
            let desc = unlit_pipeline_desc(&*shader_compiler, vertex_format, *polygon_mode)?;
            renderer.create_gfx_pipeline(desc, &frame_data.main_render_pass)?
        }
    };

    Ok(pipe)
}

fn create_renderable(
    renderer: &mut Renderer,
    world: &World,
    mesh: &Mesh,
    material: &GpuMaterial,
) -> RenderableMaterial {
    log::trace!("Creating renderable: {:?}", material);
    let material_descriptor_set = create_material_descriptor_set(renderer, material);
    let gfx_pipeline =
        get_pipeline_for(renderer, world, mesh, material, true).expect("Failed to get pipeline");
    let shadow_pipeline = if let material::GpuMaterial::PBR { .. } = material {
        Some(
            light::get_shadow_pipeline_for(renderer, world, mesh)
                .expect("Failed to create shadow pipeline"),
        )
    } else {
        None
    };

    RenderableMaterial {
        gfx_pipeline,
        shadow_pipeline,
        material_descriptor_set,
    }
}

#[profiling::function]
fn create_renderables(renderer: &mut Renderer, world: &mut World) {
    use specs::storage::StorageEntry;

    let meshes = world.read_storage::<Mesh>();
    let materials = world.read_storage::<GpuMaterial>();
    let mut should_reload = world.write_storage::<ReloadMaterial>();
    let mut renderables = world.write_storage::<RenderableMaterial>();
    let entities = world.entities();

    if !should_reload.is_empty() {
        log::info!(
            "Reloading shaders, {} entities have the ReloadMaterial tag",
            should_reload.count()
        );
    }

    for (ent, mesh, mat) in (&entities, &meshes, &materials).join() {
        let entry = renderables.entry(ent).expect("Failed to get entry!");
        match entry {
            StorageEntry::Occupied(mut entry) => {
                log::trace!("Using existing Renderable");
                if should_reload.contains(ent) {
                    log::trace!("Reloading shader for {:?}", ent);
                    // TODO: Destroy the previous pipeline
                    match get_pipeline_for(renderer, world, mesh, mat, false) {
                        Ok(pipeline) => entry.get_mut().gfx_pipeline = pipeline,
                        Err(e) => log::error!("Failed to compile pipeline: {}", e),
                    }
                }
            }
            StorageEntry::Vacant(entry) => {
                log::trace!("No Renderable found, creating new");
                let rend = create_renderable(renderer, world, mesh, mat);
                entry.insert(rend);
            }
        }
    }

    should_reload.clear();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawMode {
    Lit,
    Unlit,
    ShadowsOnly,
}

#[profiling::function]
fn draw_entities<'a>(world: &World, cmd_buf: &mut RenderPassEncoder<'a>, mode: DrawMode) {
    let model_matrices = world.read_storage::<ModelMatrix>();
    let meshes = world.read_storage::<Mesh>();
    let renderables = world.read_storage::<RenderableMaterial>();
    use trekant::pipeline::ShaderStage;

    let mut prev_handle: Option<Handle<GraphicsPipeline>> = None;

    // Only bind the pipeline if we need to
    let mut bind_pipeline = |enc: &mut RenderPassEncoder<'a>, handle: &Handle<GraphicsPipeline>| {
        if prev_handle.map(|h| h != *handle).unwrap_or(true) {
            enc.bind_graphics_pipeline(handle);
            prev_handle = Some(*handle);
        }
    };

    for (mesh, renderable, mtx) in (&meshes, &renderables, &model_matrices).join() {
        let (vertex_buffer, index_buffer) = match (&mesh.gpu_vertex_buffer, &mesh.gpu_index_buffer)
        {
            (GpuResource::Available(vbuf), GpuResource::Available(ibuf)) => (vbuf, ibuf),
            _ => continue,
        };

        let tfm = uniform::Model {
            model: mtx.0.into_col_array(),
            model_it: mtx.0.inverted().transposed().into_col_array(),
        };

        match (renderable, mode) {
            (
                RenderableMaterial {
                    shadow_pipeline: Some(shadow_pipeline),
                    ..
                },
                DrawMode::ShadowsOnly,
            ) => {
                bind_pipeline(cmd_buf, shadow_pipeline);
                cmd_buf
                    .bind_push_constant(shadow_pipeline, ShaderStage::VERTEX, &tfm)
                    .draw_mesh(vertex_buffer, index_buffer);
            }
            (
                RenderableMaterial {
                    gfx_pipeline,
                    material_descriptor_set,
                    ..
                },
                DrawMode::Lit,
            ) => {
                bind_pipeline(cmd_buf, gfx_pipeline);
                cmd_buf
                    .bind_shader_resource_group(1, material_descriptor_set, gfx_pipeline)
                    .bind_push_constant(gfx_pipeline, ShaderStage::VERTEX, &tfm)
                    .draw_mesh(vertex_buffer, index_buffer);
            }
            _ => (),
        }
    }
}

#[profiling::function]
pub fn draw_frame(world: &mut World, ui: &mut imgui::UIContext, renderer: &mut Renderer) {
    let cam_entity = match ecs::find_singleton_entity::<MainRenderCamera>(world) {
        None => {
            log::warn!("Did not find a camera entity, can't render");
            return;
        }
        Some(e) => e,
    };

    // TODO: Refactor! It would be nice to split the frame in:
    // 1. Data send
    // 2. Draw calls

    GpuUpload::resolve_pending(world, renderer);
    create_renderables(renderer, world);

    let main_window_extents = world.read_resource::<crate::io::MainWindow>().extents();

    let aspect_ratio = renderer.aspect_ratio();
    let mut frame = match renderer.next_frame() {
        frame @ Ok(_) => frame,
        Err(trekant::RenderError::NeedsResize(reason)) => {
            log::debug!("Resize reason: {:?}", reason);
            renderer
                .resize(main_window_extents)
                .expect("Failed to resize renderer");
            renderer.next_frame()
        }
        e => e,
    }
    .expect("Failed to get next frame");

    let reset_scissors = util::Rect2D {
        extent: main_window_extents,
        ..Default::default()
    };

    let ui_draw_commands = ui.build_ui(world, &mut frame);

    let frame_data = &*world.write_resource::<FrameResources>();

    {
        let mut cameras = world.write_storage::<Camera>();
        let transforms = world.read_storage::<Transform>();
        let camera = cameras.get_mut(cam_entity).unwrap();
        camera.aspect_ratio = aspect_ratio;
        let tfm = transforms
            .get(cam_entity)
            .expect("Camera needs a transform");

        let view_matrix = Mat4::from(*tfm).inverted();
        let view_proj = camera.proj_matrix() * view_matrix;
        let view_data = uniform::ViewData {
            view_proj: view_proj.into_col_array(),
            view_pos: [tfm.position.x, tfm.position.y, tfm.position.z, 1.0f32],
        };

        frame
            .update_uniform_blocking(&frame_data.engine_shader_resources.view_data, &view_data)
            .expect("Failed to update uniform");
    };

    let cmd_buffer = frame
        .new_command_buffer()
        .expect("Failed to create command buffer");

    let (mut cmd_buffer, shadow_pass_output) =
        light::shadow_pass(world, &mut frame, &frame_data.shadow, cmd_buffer);

    light::write_lighting_data(world, &mut frame, &frame_data, &shadow_pass_output);

    {
        let debug_renderer = world.read_resource::<debug::DebugRenderer>();
        debug_renderer.upload(&mut frame);
    }

    {
        // main render pass
        let FrameResources {
            main_render_pass,
            unlit_resources,
            pbr_resources,
            engine_shader_resources,
            ..
        } = frame_data;
        let mut main_rp = frame
            .begin_presentation_pass(cmd_buffer, main_render_pass)
            .expect("Failed to begin render pass");

        {
            // PBR
            let PBRPassResources { dummy_pipeline, .. } = &pbr_resources;
            main_rp
                .bind_graphics_pipeline(dummy_pipeline)
                .bind_shader_resource_group(
                    0u32,
                    &engine_shader_resources.desc_set,
                    dummy_pipeline,
                );
            draw_entities(world, &mut main_rp, DrawMode::Lit);
        }

        {
            // Unlit
            let UnlitPassResources { dummy_pipeline } = &unlit_resources;
            main_rp
                .bind_graphics_pipeline(dummy_pipeline)
                .bind_shader_resource_group(
                    0u32,
                    &engine_shader_resources.desc_set,
                    dummy_pipeline,
                );
            draw_entities(world, &mut main_rp, DrawMode::Unlit);
        }

        if let Some(ui_draw_commands) = ui_draw_commands {
            ui_draw_commands.record_draw_commands(&mut main_rp);
            main_rp.set_scissor(reset_scissors);
        }

        {
            let debug_renderer = world.read_resource::<debug::DebugRenderer>();
            debug_renderer.record_commands(&mut main_rp);
        }

        cmd_buffer = main_rp.end().expect("Failed to end main presentation pass");
    }

    frame.add_command_buffer(cmd_buffer);

    let frame = frame.finish();
    renderer
        .submit(frame)
        .or_else(|e| {
            if let trekant::RenderError::NeedsResize(reason) = e {
                log::info!("Resize reason: {:?}", reason);
                renderer.resize(world.read_resource::<crate::io::MainWindow>().extents())
            } else {
                Err(e)
            }
        })
        .expect("Failed to submit frame");
}

pub fn create_frame_resources(
    renderer: &mut Renderer,
    shader_compiler: &ShaderCompiler,
) -> FrameResources {
    use trekant::pipeline::ShaderStage;

    log::trace!("Creating frame gpu resources");

    let main_render_pass = renderer
        .presentation_render_pass(8)
        .expect("main render pass creation failed");

    // TODO: There is a lot of duplication here when we want to upload constant data
    // to a buffer with a blocking call. Can we do something like:
    // BufferManager.create_uniform_buffer_from<T>(T t, BufferMutability)?

    let view_data = {
        let view_data = uniform::ViewData {
            view_proj: [0.0; 16],
            view_pos: [0.0; 4],
        };
        let view_data = UniformBufferDescriptor::from_single(view_data, BufferMutability::Mutable);
        renderer.create_resource_blocking(view_data).expect("FAIL")
    };

    let shadow_resources = light::setup_shadow_resources(shader_compiler, renderer);

    let lighting_data = {
        let light_data = uniform::LightingData {
            lights: [uniform::PackedLight::default(); uniform::MAX_NUM_LIGHTS],
            ambient: [0.0; 4],
            num_lights: [0; 4],
        };
        let light_data =
            UniformBufferDescriptor::from_single(light_data, BufferMutability::Mutable);
        renderer.create_resource_blocking(light_data).expect("FAIL")
    };

    let shadow_data = {
        let shadow_data = uniform::ShadowData {
            matrices: Default::default(),
            count: [u32::MAX; 4],
        };
        let shadow_data =
            UniformBufferDescriptor::from_single(shadow_data, BufferMutability::Mutable);
        renderer
            .create_resource_blocking(shadow_data)
            .expect("FAIL")
    };

    let spotlight_textures = shadow_resources
        .spotlights
        .iter()
        .map(|x| (x.texture, true));
    let engine_shader_resource_group = PipelineResourceSet::builder(renderer)
        .add_buffer(&view_data, 0, ShaderStage::VERTEX | ShaderStage::FRAGMENT)
        .add_buffer(&shadow_data, 1, ShaderStage::VERTEX)
        .add_buffer(&lighting_data, 2, ShaderStage::FRAGMENT)
        .add_texture(
            &shadow_resources.directional.texture,
            3,
            ShaderStage::FRAGMENT,
            true,
        )
        .add_textures(spotlight_textures, 4, ShaderStage::FRAGMENT)
        // TODO: Add pointlights here
        .build();

    let pbr_resources = {
        // TODO: Share this code with get_pipeline_for?
        let vertex_format = VertexFormat::from([util::Format::FLOAT3; 2]);
        let (vert, frag) = shader::pbr_gltf::compile_default(&shader_compiler)
            .expect("Failed to compile default PBR shaders");
        let vert = ShaderDescriptor {
            debug_name: Some("dummy-pbr-vert".to_owned()),
            spirv_code: vert.data(),
        };
        let frag = ShaderDescriptor {
            debug_name: Some("dummy-pbr-frag".to_owned()),
            spirv_code: frag.data(),
        };

        let desc = GraphicsPipelineDescriptor::builder()
            .vert(vert)
            .frag(frag)
            .vertex_format(vertex_format)
            .build()
            .expect("Failed to build graphics pipeline descriptor");
        let dummy_pipeline = renderer
            .create_gfx_pipeline(desc, &main_render_pass)
            .expect("FAIL");

        PBRPassResources { dummy_pipeline }
    };

    let unlit_resources = {
        let vertex_format = VertexFormat::from(util::Format::FLOAT3);
        let desc = unlit_pipeline_desc(
            &shader_compiler,
            vertex_format,
            trekant::pipeline::PolygonMode::Line,
        )
        .expect("Failed to create descriptor for unlit dummy pipeline");
        let dummy_pipeline = renderer
            .create_gfx_pipeline(desc, &main_render_pass)
            .expect("Failed to create unlit dummy pipeline");

        UnlitPassResources { dummy_pipeline }
    };

    FrameResources {
        main_render_pass,
        engine_shader_resources: EngineShaderResources {
            view_data,
            lighting_data,
            shadow_data,
            desc_set: engine_shader_resource_group,
        },
        pbr_resources,
        unlit_resources,
        shadow: shadow_resources,
    }
}

pub fn setup_resources(world: &mut World, renderer: &mut Renderer) {
    let shader_compiler = shader::ShaderCompiler::new().expect("Failed to create shader compiler");

    let frame_resources = create_frame_resources(renderer, &shader_compiler);

    let debug_renderer = debug::DebugRenderer::new(
        &shader_compiler,
        &frame_resources.main_render_pass,
        &frame_resources.engine_shader_resources.view_data,
        renderer,
    );

    world.insert(shader_compiler);
    world.insert(debug_renderer);
    world.insert(debug::OneShotDebugUI::new());
    world.insert(renderer.loader().unwrap());
    world.insert(frame_resources);
    log::trace!("Done");
}

#[derive(Debug, Clone, Visitable)]
pub enum Pending<T1, T2> {
    Pending(T1),
    Available(T2),
}

struct GpuUpload;
impl GpuUpload {
    pub const ID: &'static str = "GpuUpload";
}

use resurs::Async;

use self::shader::ShaderCompiler;
fn map_pending_buffer_handle<BT>(
    h: &mut Pending<BufferHandle<Async<BT>>, BufferHandle<BT>>,
    old: BufferHandle<Async<BT>>,
    new: BufferHandle<BT>,
) {
    match h {
        Pending::Pending(cur) if cur.handle() == old.handle() => {
            *h = Pending::Available(BufferHandle::sub_buffer(new, cur.idx(), cur.n_elems()));
        }
        _ => (),
    }
}

impl GpuUpload {
    #[profiling::function]
    fn resolve_pending(world: &mut World, renderer: &mut Renderer) {
        use trekant::loader::HandleMapping;
        let loader = world.write_resource::<trekant::Loader>();
        let mut pending_materials = world.write_storage::<PendingMaterial>();
        let mut materials = world.write_storage::<GpuMaterial>();
        let mut meshes = world.write_storage::<Mesh>();
        let mut transfer_guard = loader.transfer(renderer);
        let mut generate_mipmaps = Vec::new();
        for mapping in transfer_guard.iter() {
            match mapping {
                // TODO: drain_filter()
                HandleMapping::UniformBuffer { old, new } => {
                    for (ent, _) in (&world.entities(), &pending_materials.mask().clone()).join() {
                        if let Some(pending) = pending_materials.get_mut(ent) {
                            let handle = match pending {
                                PendingMaterial::Unlit { color_uniform, .. } => color_uniform,
                                PendingMaterial::PBR {
                                    material_uniforms, ..
                                } => material_uniforms,
                            };
                            map_pending_buffer_handle(handle, old, new);

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
                HandleMapping::VertexBuffer { old, new } => {
                    for mesh in (&mut meshes).join() {
                        if !mesh.is_pending_gpu() {
                            continue;
                        }

                        mesh.try_consume_vertex_buffer(old, new);
                    }
                }
                HandleMapping::IndexBuffer { old, new } => {
                    for mesh in (&mut meshes).join() {
                        if !mesh.is_pending_gpu() {
                            continue;
                        }

                        mesh.try_consume_index_buffer(old, new);
                    }
                }
                HandleMapping::Texture { old, new } => {
                    for (ent, _) in (&world.entities(), &pending_materials.mask().clone()).join() {
                        if let Some(pending) = pending_materials.get_mut(ent) {
                            match pending {
                                PendingMaterial::PBR {
                                    normal_map,
                                    base_color_texture,
                                    metallic_roughness_texture,
                                    ..
                                } => {
                                    for tex in &mut [
                                        normal_map,
                                        base_color_texture,
                                        metallic_roughness_texture,
                                    ] {
                                        match tex {
                                            Some(Pending::Pending(tex_inner))
                                                if tex_inner.handle == old =>
                                            {
                                                generate_mipmaps.push(new);
                                                **tex = Some(Pending::Available(
                                                    material::TextureUse {
                                                        handle: new,
                                                        coord_set: tex_inner.coord_set,
                                                    },
                                                ));
                                            }
                                            _ => (),
                                        }
                                    }
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

        renderer
            .generate_mipmaps(&generate_mipmaps)
            .expect("Failed to generate mipmaps");
    }
}

impl<'a> System<'a> for GpuUpload {
    type SystemData = (
        WriteExpect<'a, trekant::Loader>,
        WriteStorage<'a, material::Unlit>,
        WriteStorage<'a, material::PhysicallyBased>,
        WriteStorage<'a, PendingMaterial>,
        WriteStorage<'a, GpuMaterial>,
        WriteStorage<'a, mesh::Mesh>,
        Entities<'a>,
    );

    fn run(&mut self, data: Self::SystemData) {
        use trekant::loader::ResourceLoader;

        let (
            loader,
            unlit_materials,
            physically_based_materials,
            mut pending_mats,
            gpu_materials,
            mut meshes,
            entities,
        ) = data;

        {
            // Unlit
            let mut ubuf = Vec::new();
            for (_, unlit, _, _) in
                (&entities, &unlit_materials, !&gpu_materials, !&pending_mats).join()
            {
                ubuf.push(uniform::UnlitUniformData {
                    color: unlit.color.into_array(),
                });
            }

            if !ubuf.is_empty() {
                let async_handle = loader
                    .load(UniformBufferDescriptor::from_vec(
                        ubuf,
                        BufferMutability::Immutable,
                    ))
                    .expect("Failed to load uniform buffer");
                for (i, (ent, unlit, _)) in (&entities, &unlit_materials, !&gpu_materials)
                    .join()
                    .enumerate()
                {
                    if let StorageEntry::Vacant(entry) = pending_mats.entry(ent).unwrap() {
                        entry.insert(PendingMaterial::Unlit {
                            color_uniform: Pending::Pending(BufferHandle::sub_buffer(
                                async_handle,
                                i as u32,
                                1,
                            )),
                            polygon_mode: unlit.polygon_mode,
                        });
                    }
                }
            }
        }

        {
            // Physically based
            let mut ubuf_pbr = Vec::new();
            for (_, pb_mat, _, _) in (
                &entities,
                &physically_based_materials,
                !&gpu_materials,
                !&pending_mats,
            )
                .join()
            {
                ubuf_pbr.push(uniform::PBRMaterialData {
                    base_color_factor: pb_mat.base_color_factor.into_array(),
                    metallic_factor: pb_mat.metallic_factor,
                    roughness_factor: pb_mat.roughness_factor,
                    normal_scale: pb_mat.normal_scale,
                    _padding: 0.0,
                });
            }

            let map_tex = |inp: &Option<material::TextureUse2>| -> Option<
                Pending<
                    material::TextureUse<resurs::Async<trekant::Texture>>,
                    material::TextureUse<trekant::Texture>,
                >,
            > {
                inp.as_ref().map(|tex| {
                    let handle = loader
                        .load_texture(tex.desc.clone())
                        .expect("Failed to load texture");
                    Pending::Pending(material::TextureUse {
                        coord_set: tex.coord_set,
                        handle,
                    })
                })
            };

            if !ubuf_pbr.is_empty() {
                let async_handle = loader
                    .load(UniformBufferDescriptor::from_vec(
                        ubuf_pbr,
                        BufferMutability::Immutable,
                    ))
                    .expect("Failed to load uniform buffer");
                for (i, (ent, pb_mat, _)) in
                    (&entities, &physically_based_materials, !&gpu_materials)
                        .join()
                        .enumerate()
                {
                    if let StorageEntry::Vacant(entry) = pending_mats.entry(ent).unwrap() {
                        entry.insert(PendingMaterial::PBR {
                            material_uniforms: Pending::Pending(BufferHandle::sub_buffer(
                                async_handle,
                                i as u32,
                                1,
                            )),
                            normal_map: map_tex(&pb_mat.normal_map),
                            base_color_texture: map_tex(&pb_mat.base_color_texture),
                            metallic_roughness_texture: map_tex(&pb_mat.metallic_roughness_texture),
                            has_vertex_colors: pb_mat.has_vertex_colors,
                        });
                    }
                }
            }
        }

        for mesh in (&mut meshes).join() {
            if mesh.is_available_gpu() || mesh.is_pending_gpu() {
                continue;
            }

            mesh.load_gpu(&loader);
        }
    }
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    register_module_systems!(builder, debug, geometry).with(GpuUpload, GpuUpload::ID, &[])
}

pub fn register_components(world: &mut World) {
    world.register::<geometry::Shape>();

    world.register::<RenderableMaterial>();
    world.register::<material::GpuMaterial>();
    world.register::<material::PendingMaterial>();
    world.register::<material::PhysicallyBased>();
    world.register::<material::Unlit>();

    world.register::<mesh::Mesh>();

    world.register::<light::ShadowViewer>();
    world.register::<light::ShadowMap>();
    world.register::<light::Light>();

    world.register::<debug::light::RenderLightVolume>();
    world.register::<debug::light::LightVolumeRenderer>();

    world.register::<debug::bounding_box::RenderBoundingBox>();
    world.register::<debug::bounding_box::BoundingBoxRenderer>();

    world.register::<debug::camera::DrawFrustum>();
}
