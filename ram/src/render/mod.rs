use std::num::NonZeroU8;

use shader::{ShaderCache, ShaderLocation};
use thiserror::Error;

use crate::ecs::prelude::*;

use trekant::{BufferDescriptor, BufferMutability, RenderPass};

use trekant::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError, ShaderDescriptor,
};
use trekant::pipeline_resource::PipelineResourceSet;
use trekant::resource::Handle;
use trekant::util;
use trekant::BufferHandle;
use trekant::RenderPassEncoder;
use trekant::Renderer;
use trekant::VertexFormat;

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
pub use material::TextureAssetLoader;
pub use mesh::{HostBuffer, HostIndexBuffer, HostVertexBuffer};

pub use mesh::Mesh;

use crate::camera::*;
use crate::ecs;
use crate::math::{Mat4, ModelMatrix, Transform, Vec3};
use ram_derive::Visitable;

pub struct PendingEntityResources<H> {
    map: std::collections::HashMap<H, Vec<Entity>>,
}

impl<T> Default for PendingEntityResources<T> {
    fn default() -> Self {
        Self {
            map: Default::default(),
        }
    }
}

type PendingEntityBuffers = PendingEntityResources<trekant::PendingBufferHandle>;
type PendingEntityTextures = PendingEntityResources<trekant::PendingTextureHandle>;

#[derive(Debug, Clone)]
pub struct DoneEntities(Vec<Entity>);

impl IntoIterator for DoneEntities {
    type IntoIter = std::vec::IntoIter<Entity>;
    type Item = Entity;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<H> PendingEntityResources<H>
where
    H: std::hash::Hash + std::cmp::Eq,
{
    pub fn flush(&mut self, handle: H) -> DoneEntities {
        self.try_flush(handle)
            .expect("Tried to flush the entities for a handle that is pending")
    }

    pub fn try_flush(&mut self, handle: H) -> Option<DoneEntities> {
        self.map.remove(&handle).map(DoneEntities)
    }

    pub fn push(&mut self, handle: H, entity: Entity) {
        self.map.entry(handle).or_default().push(entity)
    }
}

/// Utility for working with asynchronously uploaded gpu resources
#[derive(Debug, Clone, Visitable)]
pub enum GpuBuffer {
    Pending(trekant::PendingBufferHandle),
    Available(BufferHandle),
}

impl GpuBuffer {
    fn get_available(&self) -> Option<BufferHandle> {
        match self {
            Self::Available(a) => Some(*a),
            Self::Pending(_) => None,
        }
    }

    fn try_take(&mut self, old: trekant::PendingBufferHandle, new: trekant::BufferHandle) -> bool {
        match self {
            Self::Pending(pending) if pending.is_same_resource(&old) => {
                *self = Self::Available(trekant::BufferHandle::sub_buffer(
                    new,
                    pending.idx(),
                    pending.n_elems(),
                ));
                true
            }
            _ => false,
        }
    }
}

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
    view_data: BufferHandle,
    lighting_data: BufferHandle,
    world_to_shadow: BufferHandle,
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
    material_descriptor_set: Handle<PipelineResourceSet>,
}

#[derive(Debug, Error)]
pub enum MaterialError {
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] shader::Error),
    #[error("GLSL compiler error: {0}")]
    GlslCompiler(#[from] shader::CompilerError),
}

pub(crate) fn shader_path<P>(shader_path: P) -> shader::ShaderLocation
where
    P: AsRef<std::path::Path>,
{
    let shader_path: &std::path::Path = shader_path.as_ref();

    let mut path = std::path::PathBuf::new();
    path.push("render");
    path.push("shaders");

    for component in shader_path {
        path.push(component);
    }
    ShaderLocation::search(path)
}

#[profiling::function]
fn draw_entities<MaterialFilterComponent: Component>(
    world: &World,
    cmd_buf: &mut RenderPassEncoder<'_>,
) {
    let model_matrices = world.read_storage::<ModelMatrix>();
    let meshes = world.read_storage::<Mesh>();
    let renderables = world.read_storage::<RenderableMaterial>();
    let materials = world.read_storage::<MaterialFilterComponent>();
    use trekant::pipeline::ShaderStage;

    let mut prev_handle: Option<Handle<GraphicsPipeline>> = None;

    // Only bind the pipeline if we need to
    let mut bind_pipeline = |enc: &mut RenderPassEncoder<'_>, handle: &Handle<GraphicsPipeline>| {
        if prev_handle.map(|h| h != *handle).unwrap_or(true) {
            enc.bind_graphics_pipeline(handle);
            prev_handle = Some(*handle);
        }
    };

    for (mesh, renderable, mtx, _) in (&meshes, &renderables, &model_matrices, &materials).join() {
        let (vertex_buffer, index_buffer) = match (&mesh.gpu_vertex_buffer, &mesh.gpu_index_buffer)
        {
            (Some(GpuBuffer::Available(vbuf)), Some(GpuBuffer::Available(ibuf))) => (vbuf, ibuf),
            _ => continue,
        };

        let tfm = uniform::Model {
            model: mtx.0.into_col_array(),
            model_it: mtx.0.inverted().transposed().into_col_array(),
        };

        let RenderableMaterial {
            gfx_pipeline,
            material_descriptor_set,
        } = renderable;
        bind_pipeline(cmd_buf, gfx_pipeline);
        cmd_buf
            .bind_shader_resource_group(1, material_descriptor_set, gfx_pipeline)
            .bind_push_constant(gfx_pipeline, ShaderStage::VERTEX, &tfm)
            .draw_mesh(*vertex_buffer, *index_buffer);
    }
}

fn recreate_pipelines(renderer: &mut Renderer, world: &mut World) {
    let pipeline_service = world.read_resource::<shader::PipelineService>();
    pipeline_service.flush(renderer);
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

    {
        let loader = world.read_resource::<trekant::Loader>();
        loader.progress(renderer);
    }
    material::pre_frame(world, renderer);
    light::pre_frame(world, renderer);

    let main_window_extents = world.read_resource::<crate::io::MainWindow>().extents();

    // TODO: Isn't the aspect ratio old if we resize after...?
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
            .write_buffer_element(frame_data.engine_shader_resources.view_data, &view_data, 0)
            .expect("Failed to update uniform");
    };

    let cmd_buffer = frame
        .new_command_buffer()
        .expect("Failed to create command buffer");

    let (mut cmd_buffer, shadow_pass_output) =
        light::shadow_pass(world, &mut frame, &frame_data.shadow, cmd_buffer);

    light::write_lighting_data(world, &mut frame, frame_data, &shadow_pass_output);

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
            draw_entities::<material::PhysicallyBased>(world, &mut main_rp);
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
            draw_entities::<material::Unlit>(world, &mut main_rp);
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
    pipeline_service: &shader::PipelineService,
) -> FrameResources {
    use trekant::pipeline::ShaderStage;

    log::trace!("Creating frame gpu resources");

    let main_render_pass = renderer
        .presentation_render_pass(8)
        .expect("main render pass creation failed");

    let view_data = {
        let view_data = uniform::ViewData {
            view_proj: [0.0; 16],
            view_pos: [0.0; 4],
        };
        let view_data = BufferDescriptor::uniform_buffer(
            std::slice::from_ref(&view_data),
            BufferMutability::Mutable,
            trekant::BufferLayout::Std140,
        );
        renderer
            .create_buffer(view_data)
            .expect("Failed to create view data uniform buffer")
    };

    let shadow_resources = light::setup_shadow_resources(shader_compiler, shader_cache, renderer);

    let lighting_data = {
        let lighting_data = uniform::LightingData {
            lights: [uniform::PackedLight::default(); uniform::MAX_NUM_LIGHTS],
            ambient: [0.0; 4],
            num_lights: [0; 4],
        };
        let lighting_data = BufferDescriptor::uniform_buffer(
            std::slice::from_ref(&lighting_data),
            BufferMutability::Mutable,
            trekant::BufferLayout::Std140,
        );
        renderer
            .create_buffer(lighting_data)
            .expect("Failed to create lighting data uniform")
    };

    let world_to_shadow = {
        let data: Vec<uniform::Mat4> = vec![uniform::mat4_nan(); 256];

        let world_to_shadow = BufferDescriptor::storage_buffer(data, BufferMutability::Mutable);
        renderer
            .create_buffer(world_to_shadow)
            .expect("Failed to create buffer for shadow matrices")
    };

    // TODO: This is leaking the shadow resources a bit.
    let spotlight_textures = shadow_resources
        .spotlights
        .iter()
        .map(|x| (x.texture, true));
    let pointlight_textures = shadow_resources
        .pointlights
        .iter()
        .map(|x| (x.cube_map, false));

    let engine_shader_resource_group = PipelineResourceSet::builder(renderer)
        .add_buffer(view_data, 0, ShaderStage::VERTEX | ShaderStage::FRAGMENT)
        .add_buffer(world_to_shadow, 1, ShaderStage::FRAGMENT)
        .add_buffer(lighting_data, 2, ShaderStage::FRAGMENT)
        .add_texture(
            shadow_resources.directional.texture,
            3,
            ShaderStage::FRAGMENT,
            true,
        )
        .add_textures(spotlight_textures, 4, ShaderStage::FRAGMENT)
        .add_textures(pointlight_textures, 5, ShaderStage::FRAGMENT)
        .build();

    let pbr_resources = {
        // TODO: Share this code with get_pipeline_for?
        let vertex_format = VertexFormat::from([util::Format::FLOAT3; 2]);
        let (vert, frag) = shader::pbr_gltf::compile_default(shader_compiler, shader_cache)
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
        let dummy_pipeline =
            material::unlit::get_default_pipeline(renderer, pipeline_service, render_pass);
        UnlitPassResources { dummy_pipeline }
    };

    FrameResources {
        main_render_pass,
        engine_shader_resources: EngineShaderResources {
            view_data,
            lighting_data,
            world_to_shadow,
            desc_set: engine_shader_resource_group,
        },
        pbr_resources,
        unlit_resources,
        shadow: shadow_resources,
    }
}

pub fn setup_resources(world: &mut World, renderer: &mut Renderer) {
    let mut shader_compiler =
        shader::ShaderCompiler::new().expect("Failed to create shader compiler");

    let include_path_rel = std::path::PathBuf::from_iter(["render", "shaders", "include"]);
    let mut roots = Vec::with_capacity(3);
    match std::env::current_exe() {
        Ok(path) => {
            let path = path
                .parent()
                .unwrap_or_else(|| {
                    panic!(
                        "Executable always has a parent directory but {} doesn't",
                        path.display()
                    )
                })
                .to_path_buf()
                .join("shaders");
            roots.push(path);
        }
        Err(e) => {
            log::error!("Failed to read current executable location due to {e}. Can't load shader relative to it.");
        }
    }

    let build_out_dir = std::path::PathBuf::from(env!("OUT_DIR")).join("shaders");
    roots.push(build_out_dir);

    if let Ok(cwd) = std::env::current_dir() {
        let mut root = cwd;
        root.push("ram");
        root.push("src");
        roots.push(root);
    } else {
        log::error!("Failed to read current working directory, can't load shaders relative to it");
    }

    for root in roots {
        shader_compiler.add_shader_path(root.as_path());
        shader_compiler.add_include_path(root.as_path().join(&include_path_rel));
    }

    let mut pipeline_service = shader::PipelineService::new(shader::PipelineServiceConfig {
        live_recompile: true,
        n_threads: NonZeroU8::new(std::thread::available_parallelism().unwrap_or(2) as u8).unwrap(),
    });
    let frame_resources = create_frame_resources(renderer, &shader_compiler, &mut shader_cache);

    let debug_renderer = debug::DebugRenderer::new(
        &shader_compiler,
        &frame_resources.main_render_pass,
        frame_resources.engine_shader_resources.view_data,
        renderer,
    );

    world.insert(shader_compiler);
    world.insert(GlobalShaderCache(std::sync::Mutex::new(shader_cache)));
    world.insert(debug_renderer);
    world.insert(debug::OneShotDebugUI::new());
    world.insert(renderer.loader().unwrap());
    world.insert(frame_resources);
    world.insert(material::TextureAssetLoader::new());
    log::trace!("Done");
}

use self::shader::ShaderCompiler;

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    register_module_systems!(builder, debug, geometry, material, mesh)
}

pub fn register_components(world: &mut World) {
    world.register::<geometry::Shape>();

    world.register::<RenderableMaterial>();
    world.register::<material::PhysicallyBased>();
    world.register::<material::Unlit>();

    world.register::<mesh::Mesh>();

    world.register::<light::ShadowViewer>();
    world.register::<light::ShadowMap>();
    world.register::<light::ShadowPipeline>();
    world.register::<light::Light>();

    world.register::<debug::light::RenderLightVolume>();
    world.register::<debug::light::LightVolumeRenderer>();

    world.register::<debug::bounding_box::RenderBoundingBox>();
    world.register::<debug::bounding_box::BoundingBoxRenderer>();

    world.register::<debug::camera::DrawFrustum>();
}
