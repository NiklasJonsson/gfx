use vulkano::buffer::{
    cpu_pool::CpuBufferPoolSubbuffer, BufferAccess, BufferUsage, CpuBufferPool, ImmutableBuffer,
    TypedBufferAccess,
};
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferExecFuture, DynamicState,
};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, DescriptorSet};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format as VkFormat;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::image::{
    attachment::AttachmentImage, immutable::ImmutableImage, swapchain::SwapchainImage, Dimensions,
    ImageUsage, ImageViewAccess,
};
use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::swapchain;
use vulkano::swapchain::Surface;
use vulkano::swapchain::{
    AcquireError, ColorSpace, CompositeAlpha, PresentMode, SurfaceTransform, Swapchain,
};

use specs::world::EntitiesRes;
use specs::Component;
use vulkano::sampler::Sampler;
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture, NowFuture, SharingMode};

use vulkano::single_pass_renderpass;

use winit::window::Window;

use nalgebra_glm as glm;

use specs::prelude::*;
use specs::storage::StorageEntry;

use std::collections::HashSet;
use std::sync::Arc;

use crate::asset::storage::Handle;
use crate::camera::*;
use crate::common::*;

use crate::settings::{RenderMode, RenderSettings};

mod pipeline;
pub mod texture;

use texture::{GpuTextures, Texture, TextureAccess, Textures};

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

/*
    // TODO: Can we migrate to System? Would be nice but the VKManager monolith is noth "Send +
    // Sync". We might get by by mutexing and such but do we really need it? Might be able to use
    // thread_local_system which would not require thread safe systems.
    pub fn draw_next_frame(&mut self, world: &mut World) {
        let active_camera = world.read_resource::<ActiveCamera>();
        let positions = world.read_storage::<Position>();
        let cam_rots = world.read_storage::<CameraRotationState>();
        let renderables = world.read_storage::<OldRenderable>();
        let model_matrices = world.read_storage::<ModelMatrix>();
        let entities = world.read_resource::<EntitiesRes>();

        let camera_entity = active_camera.0.unwrap();

        let frame_idx = self.current_sc_index;

        let cam_pos = positions
            .get(camera_entity)
            .expect("Could not get position component for camera");

        let cam_rotation_state = cam_rots
            .get(camera_entity)
            .expect("Could not get rotation state for camera");

        // TODO: Camera system should write to ViewMatrixResource at the end of system
        // and we should read it here.
        let view = FreeFlyCameraController::get_view_matrix_from(cam_pos, cam_rotation_state);

        log::trace!("View matrix: {:#?}", view);

        let dims = get_physical_window_dims(self.vk_surface.window());
        let aspect_ratio = dims[0] as f32 / dims[1] as f32;
        let proj = get_proj_matrix(aspect_ratio);

        let vp_buf = pipeline::pbr::vs::base::ty::Transforms {
            view: view.into(),
            proj: proj.into(),
        };

        let transforms_sub_buf = self
            .transforms_buf
            .next(vp_buf)
            .expect("Could not get next ring buffer sub buffer for view proj");

        let lighting_data = pipeline::pbr::fs::base::ty::LightingData {
            light_pos: [5.0f32, 5.0f32, 5.0f32],
            view_pos: cam_pos.xyz().into(),
            _dummy0: [0; 4], // Magic Vulkano alignment
        };

        let lighting_sub_buf = self
            .lighting_data_buf
            .next(lighting_data)
            .expect("Could not get next ring buffer sub buffer for view proj");

        let prev_frame = std::mem::replace(&mut self.frame_completions[frame_idx], None).unwrap();

        let clear_color = vec![
            [0.0, 0.0, 0.0, 1.0].into(),
            vulkano::format::ClearValue::None,
            1.0f32.into(),
        ];

        // Render all the renderables as one render pass
        let mut builder_orig = AutoCommandBufferBuilder::primary_one_time_submit(
            Arc::clone(&self.vk_device),
            self.graphics_queue.family(),
        )
        .expect("Failed to create command buffer builder");

        let builder = builder_orig
            .begin_render_pass(
                Arc::clone(&self.framebuffers[frame_idx]),
                false,
                clear_color,
            )
            .expect("Failed after begin render pass");

        let model_ubo_buf = CpuBufferPool::uniform_buffer(Arc::clone(&self.vk_device));

        let builder =
            (&entities, &renderables)
                .join()
                .fold(builder, |builder, (ent, OldRenderable)| {
                    OldRenderable.record_draw_commands(
                        builder,
                        &transforms_sub_buf,
                        &lighting_sub_buf,
                        &model_ubo_buf,
                        model_matrices
                            .get(ent)
                            .copied()
                            .unwrap_or_else(ModelMatrix::identity),
                    );

                    builder
                });

        builder
            .end_render_pass()
            .expect("Unable to end render pass");

        let cmd_buf = builder_orig.build().expect("Unable to build render pass");

        let presented = prev_frame
            .then_execute(Arc::clone(&self.graphics_queue), cmd_buf)
            .expect("unable to execute render cmd buf")
            .then_swapchain_present(
                Arc::clone(&self.graphics_queue),
                Arc::clone(&self.swapchain),
                frame_idx,
            )
            .then_signal_fence_and_flush();

        let rendered_and_presented = match presented {
            Ok(r) => r,
            Err(FlushError::OutOfDate) => {
                self.recreate_swap_chain();
                return;
            }
            Err(e) => panic!(
                "Can't write to the swapchain image (idx: {}):\n\t{}",
                frame_idx, e
            ),
        };

        self.frame_completions[frame_idx] = Some(Box::new(rendered_and_presented));
    }
*/

fn get_proj_matrix(aspect_ratio: f32) -> glm::Mat4 {
    // TODO: Rewrite this
    let mut proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0);

    // glm::perspective is based on opengl left-handed coordinate system, vulkan has the y-axis
    // inverted (right-handed upside-down).
    proj[(1, 1)] *= -1.0;

    proj
}

use trekanten::{Renderer, Frame};
use trekanten::command;
use trekanten::resource::Handle;
use trekanten::mesh;
use trekanten::pipeline;
use trekanten::descriptor;
use trekanten::uniform;

pub struct Mesh {
    vertex_buffer: Handle<mesh::VertexBuffer>,
    index_buffer: Handle<mesh::IndexBuffer>,
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct RenderableMaterial {
    gfx_pipeline: Handle<pipeline::GraphicsPipeline>,
    material_descriptor_set: Handle<descriptor::DescriptorSet>,
    mode: RenderMode,
}

fn create_material_descriptor_set2(renderer: &Renderer, material: &Material) -> Handle<descriptor::DescriptorSet> {
    match material {
        Material::PBR {
            material_uniforms,
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
        } => {

            let mut desc_set_builder = descriptor::DescriptorSet::builder(renderer)
            .add_buffer(material_uniforms);
            
            if let Some(nm) = normal_map {
                desc_set_builder.add_texture(&nm.tex.handle);
            }

            if let Some(bct) = base_color_texture {
                desc_set_builder.add_texture(&bct.handle);
            }

            if let Some(mrt) = metallic_roughness_texture {
                desc_set_builder.add_texture(&mrt.handle);
            }

            desc_set_builder.build()
        },
        _ => unimplemented!(),
    }
}

fn create_renderable(renderer: &mut Renderer, material: &Material, render_mode: &RenderMode) -> RenderableMaterial
{
    log::trace!("Creating renderable: {:?}, {:?}", material, render_mode);
    let material_descriptor_set = create_material_descriptor_set2(renderer, material);
    let gfx_pipeline = pipeline::get_pipeline_for(&material, render_mode);
    RenderableMaterial {
        gfx_pipeline,
        material_descriptor_set,
        mode: *render_mode,
    }
}

fn draw_model(renderer: &Renderer, cmd_buf: command::CommandBuffer, renderable: &RenderableMaterial, mesh: &Mesh) -> command::CommandBuffer {
{
    // TODO: Call functions on frame instead of directly on the frame buffer
    let gfx_pipeline = renderer
        .get_resource(&renderable.gfx_pipeline)
        .expect("Missing graphics pipeline");
    let index_buffer = renderer
        .get_resource(&mesh.index_buffer)
        .expect("Missing index buffer");
    let vertex_buffer = renderer
        .get_resource(&mesh.vertex_buffer)
        .expect("Missing vertex buffer");
    let desc_set = renderer
        .get_descriptor_set(&renderable.material_descriptor_set)
        .expect("Missing descriptor set");

    // TODO: Transform as push constant

    unimplemented!()

    /*
    cmd_buf
    .bind_graphics_pipeline(&gfx_pipeline)
    .bind_descriptor_set(&desc_set, &gfx_pipeline)
    .bind_index_buffer(&index_buffer)
    .bind_vertex_buffer(&vertex_buffer)
    // TODO: How many indices? Get it from the handle?
    .draw_indexed(0 as u32)
    */

}

pub fn draw_frame(world: &mut World, renderer: &mut Renderer) {
    let meshes = world.read_storage::<Mesh>();
    let materials = world.read_storage::<Material>();
    let mut renderables = world.write_storage::<RenderableMaterial>();

    let entities = world.read_resource::<EntitiesRes>();
    let mut render_settings = world.write_resource::<RenderSettings>();
    let cpu_textures = world.read_resource::<Textures>();

    let render_mode = render_settings.render_mode;
    let reload_shaders = render_settings.reload_runtime_shaders;

    if reload_shaders {
        log::debug!("Shader reload requested");
    }

    let frame = renderer.next_frame().expect("Failed to get frame");
    let render_pass = renderer.render_pass();
    let extent = renderer.swapchain_extent();
    let framebuffer = renderer.framebuffer(&frame);
    let mut cmd_buf = frame
        .new_command_buffer()
        .expect("Failed to create new command buffer")
        .begin_render_pass(render_pass, framebuffer, extent);

    // TODO: Bind lighting data, view & proj matrix

    for (ent, mesh, mat) in (&entities, &meshes, &materials).join() {
        // TODO: Move to function
        let entry = renderables.entry(ent).expect("Failed to get entry!");
        let renderable = match entry {
            StorageEntry::Occupied(mut occ_entry) => {
                if occ_entry.get().mode != render_mode {
                    log::trace!("Renderable did not match render mode, creating new");
                    unimplemented!()
                    /*
                    let rend = create_renderable(&renderer, mat, render_mode);
                    occ_entry.insert(rend)
                    */
                } else {
                    log::trace!("Using existing Renderable");
                    if reload_shaders {
                        log::trace!("Reloading shader");
                        if let ShaderUse::RunTime { .. } = &mat.compilation_mode {
                            unimplemented!()
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
            }
            StorageEntry::Vacant(vac_entry) => {
                log::trace!("No Renderable found, creating new");
                let rend = self.create_renderable(&cpu_textures, mesh, mat, render_mode);
                vac_entry.insert(rend)
            }
        };
        draw_mode(renderer, cmd_buf, renderable, mesh);
    }
    .end_render_pass()
    .end()
    .expect("Failed to end command buffer");

    frame.add_command_buffer(cmd_buf);
    renderer.submit(frame).expect("Failed to submit frame");
    render_settings.reload_runtime_shaders = false;
}
