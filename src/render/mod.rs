use specs::world::EntitiesRes;
use specs::Component;

use std::path::PathBuf;

use nalgebra_glm as glm;

use specs::prelude::*;
use specs::storage::StorageEntry;

use trekanten::command;
use trekanten::descriptor::DescriptorSet;
use trekanten::mesh::VertexBuffer;
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
pub mod uniform;

use crate::camera::*;
use crate::common::{Material, ModelMatrix, Position, ShaderUse};

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
}

#[derive(Default)]
pub struct FrameData {
    pub light_buffer: BufferHandle<UniformBuffer>,
    pub frame_set: Handle<DescriptorSet>,
    pub transforms_buffer: BufferHandle<UniformBuffer>,
    pub dummy_pipeline: Handle<GraphicsPipeline>,
}

fn get_view_data(world: &World) -> (glm::Mat4, Position) {
    let camera_entity = world
        .read_resource::<ActiveCamera>()
        .0
        .expect("No active camera!");

    let positions = world.read_storage::<Position>();

    let cam_pos = positions
        .get(camera_entity)
        .expect("Could not get position component for camera");

    let rots = world.read_storage::<CameraRotationState>();
    let cam_rotation_state = rots
        .get(camera_entity)
        .expect("Could not get rotation state for camera");

    // TODO: Camera system should write to ViewMatrixResource at the end of system
    // and we should read it here.
    let view = FreeFlyCameraController::get_view_matrix_from(cam_pos, cam_rotation_state);
    log::trace!("View matrix: {:#?}", view);

    (view, *cam_pos)
}

fn get_proj_matrix(aspect_ratio: f32) -> glm::Mat4 {
    let mut proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0);

    // glm::perspective is based on opengl left-handed coordinate system, vulkan has the y-axis
    // inverted (right-handed upside-down).
    proj[(1, 1)] *= -1.0;

    proj
}

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
    match &material.data {
        material::MaterialData::PBR {
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

pub fn get_pipeline_for(
    renderer: &mut Renderer,
    world: &World,
    mesh: &Mesh,
    mat: &material::MaterialData,
) -> Result<Handle<GraphicsPipeline>, PipelineError> {
    let vbuf: &VertexBuffer = renderer
        .get_resource(mesh.vertex_buffer.handle())
        .expect("Invalid handle");
    let vertex_format = vbuf.format.clone();
    let shaders = world.read_resource::<pipeline::PrecompiledShaders>();
    let pipe = match mat {
        material::MaterialData::PBR {
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
            let def = pipeline::ShaderDefinition {
                has_tex_coords: has_nm || has_bc || has_mr,
                has_vertex_colors: *has_vertex_colors,
                has_tangents: has_nm,
                has_base_color_texture: has_bc,
                has_metallic_roughness_texture: has_mr,
                has_normal_map: has_nm,
            };
            let (vert, frag) = (shaders.get_vert(&def), shaders.get_frag(&def));
            let desc = GraphicsPipelineDescriptor {
                vert: ShaderDescriptor::FromPath(PathBuf::from(vert)),
                frag: ShaderDescriptor::FromPath(PathBuf::from(frag)),
                vertex_format,
            };

            renderer
                .create_resource(desc)
                .expect("Failed to create pipeline")
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
        get_pipeline_for(renderer, world, mesh, &material.data).expect("Failed to get pipeline");
    RenderableMaterial {
        gfx_pipeline,
        material_descriptor_set,
        mode: render_mode,
    }
}

fn draw_model(
    renderer: &Renderer,
    cmd_buf: command::CommandBuffer,
    renderable: &RenderableMaterial,
    mesh: &Mesh,
    mtx: &ModelMatrix,
) -> command::CommandBuffer {
    let gfx_pipeline = renderer
        .get_resource(&renderable.gfx_pipeline)
        .expect("Missing graphics pipeline");
    let index_buffer = renderer
        .get_resource(mesh.index_buffer.handle())
        .expect("Missing index buffer");
    let vertex_buffer = renderer
        .get_resource(mesh.vertex_buffer.handle())
        .expect("Missing vertex buffer");
    let mat_desc_set = renderer
        .get_descriptor_set(&renderable.material_descriptor_set)
        .expect("Missing descriptor set");

    let vertex_index = mesh.vertex_buffer.idx();
    let indices_index = mesh.index_buffer.idx();
    let n_indices = mesh.index_buffer.len();

    let trn = uniform::Model {
        model: mtx.0.into(),
        model_it: glm::inverse_transpose(mtx.0).into(),
    };

    cmd_buf
        .bind_graphics_pipeline(&gfx_pipeline)
        .bind_descriptor_set(1, &mat_desc_set, &gfx_pipeline)
        .bind_index_buffer(&index_buffer, indices_index)
        .bind_vertex_buffer(&vertex_buffer, vertex_index)
        .bind_push_constant(
            &gfx_pipeline,
            trekanten::pipeline::ShaderStage::Vertex,
            &trn,
        )
        .draw_indexed(n_indices as u32)
}

fn draw_entities(
    renderer: &mut Renderer,
    world: &mut World,
    mut cmd_buf: command::CommandBuffer,
    render_mode: RenderMode,
    reload_shaders: bool,
) -> command::CommandBuffer {
    let model_matrices = world.read_storage::<ModelMatrix>();
    let meshes = world.read_storage::<Mesh>();
    let materials = world.read_storage::<Material>();
    let mut renderables = world.write_storage::<RenderableMaterial>();
    let entities = world.read_resource::<EntitiesRes>();

    for (ent, mesh, mat, mtx) in (&entities, &meshes, &materials, &model_matrices).join() {
        // TODO: Move to function
        let entry = renderables.entry(ent).expect("Failed to get entry!");
        cmd_buf = match entry {
            StorageEntry::Occupied(occ_entry) => {
                if occ_entry.get().mode != render_mode {
                    log::trace!("Renderable did not match render mode, creating new");
                    todo!("No support for render modes yet!")
                /*
                let rend = create_renderable(&renderer, world, mat, render_mode);
                occ_entry.insert(rend)
                */
                } else {
                    log::trace!("Using existing Renderable");
                    if reload_shaders {
                        log::trace!("Reloading shader");
                        if let ShaderUse::Reloadable { .. } = &mat.compilation_mode {
                            todo!("No support for reloadable shaders yet")
                            /* TODO
                            occ_entry.get_mut().g_pipeline = pipeline::create_graphics_pipeline(
                                &self.vk_device,
                                &self.render_pass,
                                self.swapchain.dimensions(),
                                render_mode.into(),
                                &mesh.vertex_data,
                                &mat.data,
                                &mat.compilation_mode,
                            )
                            */
                        }
                    }
                }
                draw_model(renderer, cmd_buf, occ_entry.get(), mesh, mtx)
            }
            StorageEntry::Vacant(vac_entry) => {
                log::trace!("No Renderable found, creating new");
                let rend = create_renderable(renderer, world, mesh, mat, render_mode);
                let cmd_buf = draw_model(renderer, cmd_buf, &rend, mesh, mtx);
                vac_entry.insert(rend);
                cmd_buf
            }
        };
    }

    cmd_buf
}

pub fn draw_frame(world: &mut World, renderer: &mut Renderer) {
    let (render_mode, reload_shaders) = {
        let render_settings = world.read_resource::<RenderSettings>();
        (
            render_settings.render_mode,
            render_settings.reload_runtime_shaders,
        )
    };

    if reload_shaders {
        log::debug!("Shader reload requested");
    }
    // Note: Don't do any rendering related stuff before this.
    // TODO: Move functions from renderer to frame to ensure ordering.
    let mut frame = renderer.next_frame().expect("Failed to get frame");

    // Per-frame data
    let mut cmd_buf = {
        let FrameData {
            light_buffer,
            frame_set,
            transforms_buffer,
            dummy_pipeline,
        } = &*world.read_resource::<FrameData>();

        let (view_matrix, cam_pos) = get_view_data(world);

        let lighting_data = uniform::LightingData {
            light_pos: [5.0f32, 5.0f32, 5.0f32, 0.0f32],
            view_pos: [cam_pos.x(), cam_pos.y(), cam_pos.z(), 0.0f32],
        };

        let transforms = uniform::Transforms {
            view: view_matrix.into(),
            proj: get_proj_matrix(renderer.aspect_ratio()).into(),
        };
        renderer
            .update_uniform(light_buffer, &lighting_data)
            .expect("Failed to update uniform");
        renderer
            .update_uniform(transforms_buffer, &transforms)
            .expect("Failed to update uniform");

        let render_pass = renderer.render_pass();
        let extent = renderer.swapchain_extent();
        let framebuffer = renderer.framebuffer(&frame);
        let frame_set = renderer
            .get_descriptor_set(&frame_set)
            .expect("Missing descriptor set");

        let dummy_pipeline = renderer
            .get_resource(dummy_pipeline)
            .expect("Missing pipeline!");

        frame
            .new_command_buffer()
            .expect("Failed to create new command buffer")
            .begin_render_pass(render_pass, framebuffer, extent)
            .bind_descriptor_set(0, frame_set, dummy_pipeline)
    };

    cmd_buf = draw_entities(renderer, world, cmd_buf, render_mode, reload_shaders);

    cmd_buf = cmd_buf
        .end_render_pass()
        .end()
        .expect("Failed to end command buffer");

    frame.add_command_buffer(cmd_buf);
    renderer.submit(frame).expect("Failed to submit frame");
    world
        .write_resource::<RenderSettings>()
        .reload_runtime_shaders = false;
}

pub fn setup_resources(world: &mut World, mut renderer: &mut Renderer) {
    let shaders = pipeline::PrecompiledShaders::new();

    log::trace!("Creating dummy pipeline");
    let desc = UniformBufferDescriptor::Uninitialized::<uniform::LightingData> { n_elems: 1 };
    let light_buffer = renderer.create_resource(desc).expect("FAIL");

    let desc = UniformBufferDescriptor::Uninitialized::<uniform::Transforms> { n_elems: 1 };
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

    let (vert, frag) = shaders.get_default();
    let desc = GraphicsPipelineDescriptor {
        vert: ShaderDescriptor::FromPath(PathBuf::from(vert)),
        frag: ShaderDescriptor::FromPath(PathBuf::from(frag)),
        vertex_format,
    };

    let dummy_pipeline = renderer.create_resource(desc).expect("FAIL");

    world.insert(FrameData {
        light_buffer,
        frame_set,
        transforms_buffer,
        dummy_pipeline,
    });

    world.insert(shaders);
}
