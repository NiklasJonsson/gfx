use specs::world::EntitiesRes;
use specs::Component;

use thiserror::Error;

use specs::prelude::*;
use specs::storage::StorageEntry;

use trekanten::command;
use trekanten::descriptor::DescriptorSet;
use trekanten::mesh::{BufferMutability, VertexBuffer};
use trekanten::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError, ShaderDescriptor,
};
use trekanten::resource::Handle;
use trekanten::resource::ResourceManager;
use trekanten::uniform::{UniformBuffer, UniformBufferDescriptor};
use trekanten::util;
use trekanten::vertex::VertexFormat;
use trekanten::BufferHandle;
use trekanten::Renderer;

pub mod material;
pub mod pipeline;
pub mod ui;
pub mod uniform;

use crate::camera::*;
use crate::math::{Mat4, ModelMatrix, Transform, Vec3};
use material::Material;

use crate::settings::{RenderMode, RenderSettings};

#[derive(Debug, Default)]
pub struct ActiveCamera(Option<Entity>);

impl ActiveCamera {
    pub fn empty() -> Self {
        ActiveCamera(None)
    }

    pub fn with_entity(entity: Entity) -> Self {
        ActiveCamera(Some(entity))
    }

    pub fn camera_pos(world: &World) -> Vec3 {
        let camera_entity = world
            .read_resource::<ActiveCamera>()
            .0
            .expect("No active camera!");

        let transforms = world.read_storage::<Transform>();

        transforms
            .get(camera_entity)
            .expect("Could not get position component for camera")
            .position
    }
}

#[derive(Default)]
pub struct FrameData {
    pub light_buffer: BufferHandle<UniformBuffer>,
    pub frame_set: Handle<DescriptorSet>,
    pub transforms_buffer: BufferHandle<UniformBuffer>,
    pub dummy_pipeline: Handle<GraphicsPipeline>,
}

fn get_view_data(world: &World) -> (Mat4, Vec3) {
    let cam_pos = ActiveCamera::camera_pos(world);

    let camera_entity = world
        .read_resource::<ActiveCamera>()
        .0
        .expect("No active camera!");
    let rots = world.read_storage::<CameraRotationState>();
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
#[storage(NullStorage)]
pub struct ReloadMaterial;

#[derive(Component)]
#[storage(VecStorage)]
pub struct Mesh(pub trekanten::mesh::Mesh);

impl std::ops::Deref for Mesh {
    type Target = trekanten::mesh::Mesh;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Into<Mesh> for trekanten::mesh::Mesh {
    fn into(self) -> Mesh {
        Mesh(self)
    }
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct RenderableMaterial {
    gfx_pipeline: Handle<GraphicsPipeline>,
    material_descriptor_set: Handle<DescriptorSet>,
    mode: RenderMode,
}

// TODO: Bindings here need to match with shader
fn create_material_descriptor_set(
    renderer: &mut Renderer,
    material: &Material,
) -> Handle<DescriptorSet> {
    match &material {
        material::Material::PBR {
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
                trekanten::pipeline::ShaderStage::Fragment,
            );

            if let Some(bct) = &base_color_texture {
                desc_set_builder = desc_set_builder.add_texture(
                    &bct.handle,
                    1,
                    trekanten::pipeline::ShaderStage::Fragment,
                );
            }

            if let Some(mrt) = &metallic_roughness_texture {
                desc_set_builder = desc_set_builder.add_texture(
                    &mrt.handle,
                    2,
                    trekanten::pipeline::ShaderStage::Fragment,
                );
            }

            if let Some(nm) = &normal_map {
                desc_set_builder = desc_set_builder.add_texture(
                    &nm.tex.handle,
                    3,
                    trekanten::pipeline::ShaderStage::Fragment,
                );
            }

            desc_set_builder.build()
        }
        _ => unimplemented!("Could not create descriptor set, unsupported material"),
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
    mesh: &Mesh,
    mat: &material::Material,
) -> Result<Handle<GraphicsPipeline>, MaterialError> {
    let vbuf: &VertexBuffer = renderer
        .get_resource(&mesh.vertex_buffer)
        .expect("Invalid handle");
    let vertex_format = vbuf.format().clone();
    let shader_compiler = world.read_resource::<pipeline::ShaderCompiler>();
    let pipe = match mat {
        material::Material::PBR {
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
                .build()?;

            renderer.create_resource(desc)?
        }
        m => todo!("No support for this material yet {:?}", m),
    };

    Ok(pipe)
}

fn create_renderable(
    renderer: &mut Renderer,
    world: &World,
    mesh: &Mesh,
    material: &Material,
    render_mode: RenderMode,
) -> RenderableMaterial {
    log::trace!("Creating renderable: {:?}, {:?}", material, render_mode);
    let material_descriptor_set = create_material_descriptor_set(renderer, material);
    let gfx_pipeline =
        get_pipeline_for(renderer, world, mesh, &material).expect("Failed to get pipeline");
    RenderableMaterial {
        gfx_pipeline,
        material_descriptor_set,
        mode: render_mode,
    }
}

fn create_renderables(renderer: &mut Renderer, world: &mut World, render_mode: RenderMode) {
    let meshes = world.read_storage::<Mesh>();
    let materials = world.read_storage::<Material>();
    let mut should_reload = world.write_storage::<ReloadMaterial>();
    let mut renderables = world.write_storage::<RenderableMaterial>();
    let entities = world.read_resource::<EntitiesRes>();

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
                        match get_pipeline_for(renderer, world, mesh, mat) {
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

fn draw_entities<'a>(world: &World, cmd_buf: &mut command::CommandBufferBuilder<'a>) {
    let model_matrices = world.read_storage::<ModelMatrix>();
    let meshes = world.read_storage::<Mesh>();
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
                trekanten::pipeline::ShaderStage::Vertex,
                &trn,
            )
            .draw_mesh(&mesh);
    }
}

pub fn draw_frame(world: &mut World, ui: &mut ui::UIContext, renderer: &mut Renderer) {
    let (render_mode, light_pos) = {
        let render_settings = world.read_resource::<RenderSettings>();
        (render_settings.render_mode, render_settings.light_pos)
    };

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

    let (view_matrix, cam_pos) = get_view_data(world);
    let lighting_data = uniform::LightingData {
        light_pos: [light_pos.x, light_pos.y, light_pos.z, 1.0f32],
        view_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0f32],
    };
    let transforms = uniform::Transforms {
        view: view_matrix.into_col_arrays(),
        proj: get_proj_matrix(aspect_ratio).into_col_arrays(),
    };

    let ui_draw_commands = ui.build_ui(world, &mut frame);

    let FrameData {
        light_buffer,
        frame_set,
        transforms_buffer,
        dummy_pipeline,
    } = &*world.read_resource::<FrameData>();

    frame
        .update_uniform_blocking(light_buffer, &lighting_data)
        .expect("Failed to update uniform");
    frame
        .update_uniform_blocking(transforms_buffer, &transforms)
        .expect("Failed to update uniform");

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
    let shader_compiler =
        pipeline::ShaderCompiler::new().expect("Failed to create shader compiler");

    log::trace!("Creating dummy pipeline");
    let data = [uniform::LightingData {
        light_pos: [0.0; 4],
        view_pos: [0.0; 4],
    }];
    let desc = UniformBufferDescriptor {
        data: &data,
        mutability: BufferMutability::Mutable,
    };
    let light_buffer = renderer.create_resource(desc).expect("FAIL");

    let data = [uniform::Transforms {
        view: [[0.0; 4]; 4],
        proj: [[0.0; 4]; 4],
    }];
    let desc = UniformBufferDescriptor {
        data: &data,
        mutability: BufferMutability::Mutable,
    };

    let transforms_buffer = renderer.create_resource(desc).expect("FAIL");

    let frame_set = DescriptorSet::builder(&mut renderer)
        .add_buffer(
            &transforms_buffer,
            0,
            trekanten::pipeline::ShaderStage::Vertex,
        )
        .add_buffer(&light_buffer, 1, trekanten::pipeline::ShaderStage::Fragment)
        .build();

    let vertex_format = VertexFormat::builder()
        .add_attribute(util::Format::FLOAT3)
        .add_attribute(util::Format::FLOAT3)
        .build();

    let (vert, frag) =
        pipeline::pbr_gltf::compile_default(&shader_compiler).expect("Failed to compile");
    let desc = GraphicsPipelineDescriptor::builder()
        .vert(ShaderDescriptor::FromRawSpirv(vert.data()))
        .frag(ShaderDescriptor::FromRawSpirv(frag.data()))
        .vertex_format(vertex_format)
        .build()
        .expect("Failed to build graphics pipeline descriptor");

    let dummy_pipeline = renderer.create_resource(desc).expect("FAIL");

    world.insert(FrameData {
        light_buffer,
        frame_set,
        transforms_buffer,
        dummy_pipeline,
    });

    world.insert(shader_compiler);
    log::trace!("Done");
}

pub fn register_components(world: &mut World) {
    // Register all component types
    world.register::<RenderableMaterial>();
    world.register::<Mesh>();
    world.register::<Material>();
    world.register::<crate::graph::Children>();
    world.register::<crate::graph::Parent>();
    world.register::<crate::camera::Camera>();
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        AssignReloadMaterials,
        "assign_reload_materials",
        &[crate::settings::RENDER_SETTINGS_SYS_ID],
    )
}

struct AssignReloadMaterials;

impl<'a> System<'a> for AssignReloadMaterials {
    type SystemData = (
        Write<'a, RenderSettings>,
        Read<'a, EntitiesRes>,
        ReadStorage<'a, Mesh>,
        ReadStorage<'a, Material>,
        WriteStorage<'a, ReloadMaterial>,
    );

    fn run(
        &mut self,
        (mut render_settings, entities, meshes, materials, mut reloads): Self::SystemData,
    ) {
        let should_reload = render_settings.reload_shaders;

        if !should_reload {
            return;
        }

        for (ent, _mesh, _mat) in (&entities, &meshes, &materials).join() {
            reloads
                .insert(ent, ReloadMaterial {})
                .expect("Failed to insert");
        }

        render_settings.reload_shaders = false;
    }
}
