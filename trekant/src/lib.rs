pub use ash::vk;

mod buffer;
mod loader;
pub mod pipeline;
pub mod pipeline_resource;
pub mod resource;
pub mod std140;
pub mod traits;
pub mod util;
pub mod vertex;

mod backend;
mod common;
mod descriptor;
mod error;
mod render_pass;
mod render_target;
mod texture;

pub use backend::command::CommandBuffer;
pub use buffer::{
    AsyncBufferHandle, BufferDescriptor, BufferHandle, BufferLayout, BufferMutability, BufferType,
    DeviceBuffer, HostBuffer, HostIndexBuffer, HostStorageBuffer, HostUniformBuffer,
    HostVertexBuffer, IndexBufferType, StorageBufferType, UniformBufferType, VertexBufferType,
};
pub use descriptor::DescriptorData;
pub use error::RenderError;
pub use error::ResizeReason;
pub use loader::{HandleMapping, Loader, LoaderError};
pub use pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError, ShaderStage};
pub use pipeline_resource::PipelineResourceSet;
pub use render_pass::{RenderPass, RenderPassEncoder};
pub use render_target::RenderTarget;
pub use resource::{Async, Handle};
pub use std140::{Std140, Std140Struct};
pub use texture::{
    BorderColor, Filter, MipMaps, SamplerAddressMode, SamplerDescriptor, Texture,
    TextureDescriptor, TextureType, TextureUsage,
};
pub use traits::{PushConstant, Uniform};
pub use trekant_derive::Std140;
pub use util::{Extent2D, Format};
pub use vertex::VertexFormat;

use crate::backend::image::ImageDescriptor;
use crate::backend::vk::{image::Image, MemoryError};
use backend::device::HasVkDevice as _;
use backend::{command, device, framebuffer, instance, surface, swapchain, sync};
use common::MAX_FRAMES_IN_FLIGHT;
use texture::TextureError;

// Notes:
// We can have N number of swapchain images, it depends on the backing presentation implementation.
// Generally, we are aiming for three images + MAILBOX (render one and use the latest of the two waiting)
//
// We use MAX_FRAMES_IN_FLIGHT (2, hardcoded atm) frames in flight at once. This allows us to start the next frame directly after we render.
// Whenever next_frame() is called, it can be thought of as binding one of the two frames to a particular swapchain image.
// All rendering in that frame will be done on that swapchain image/framebuffer.

struct FrameSynchronization {
    image_available: sync::Semaphore,
    render_done: sync::Semaphore,
    in_flight: sync::Fence,
    command_pool: Option<command::CommandPool>,
}

impl FrameSynchronization {
    pub fn new(device: &device::Device) -> Result<Self, sync::SyncError> {
        let image_avail = sync::Semaphore::new(device)?;
        let render_done = sync::Semaphore::new(device)?;
        let in_flight = sync::Fence::signaled(device)?;

        Ok(Self {
            image_available: image_avail,
            render_done,
            in_flight,
            command_pool: None,
        })
    }
}

pub struct Frame<'a> {
    renderer: &'a mut Renderer,
    recorded_command_buffers: Vec<vk::CommandBuffer>,
    gfx_command_pool: command::CommandPool,
}

pub struct FinishedFrame {
    recorded_command_buffers: Vec<vk::CommandBuffer>,
    gfx_command_pool: command::CommandPool,
}

impl<'a> Frame<'a> {
    pub fn new_command_buffer(&self) -> Result<command::CommandBuffer, command::CommandError> {
        self.gfx_command_pool
            .create_command_buffer(command::CommandBufferSubmission::Single)
    }

    pub fn add_command_buffer(&mut self, mut cmd_buffer: command::CommandBuffer) {
        cmd_buffer.end().expect("Failed to end command buffer");
        self.recorded_command_buffers
            .push(*cmd_buffer.vk_command_buffer());
    }
}

impl<'a> Frame<'a> {
    pub fn get_buffer(&self, handle: BufferHandle) -> Option<&DeviceBuffer> {
        self.renderer.get_buffer(handle)
    }

    pub fn write_buffer_at<T: bytemuck::Pod>(
        &mut self,
        h: BufferHandle,
        start: usize,
        data: &[T],
    ) -> Result<(), RenderError> {
        self.renderer.write_buffer_at(h, start, data)
    }

    pub fn write_buffer<T: bytemuck::Pod>(
        &mut self,
        h: BufferHandle,
        data: &[T],
    ) -> Result<(), RenderError> {
        self.renderer.write_buffer(h, data)
    }

    // idx is in terms of T
    pub fn write_buffer_element<T: bytemuck::Pod>(
        &mut self,
        h: BufferHandle,
        data: &T,
        idx: usize,
    ) -> Result<(), RenderError> {
        self.renderer.write_buffer_element(h, data, idx)
    }

    pub fn get_texture(&self, handle: &Handle<Texture>) -> Option<&Texture> {
        self.renderer.get_texture(handle)
    }

    pub fn create_buffer(
        &mut self,
        descriptor: BufferDescriptor<'_>,
    ) -> Result<BufferHandle, MemoryError> {
        self.renderer.create_buffer(descriptor)
    }

    pub fn create_buffer_with(
        &mut self,
        descriptor: BufferDescriptor<'_>,
        prev: BufferHandle,
    ) -> Result<BufferHandle, MemoryError> {
        self.renderer.create_buffer_with(descriptor, prev)
    }

    pub fn create_buffer_ext(
        &mut self,
        descriptor: BufferDescriptor<'_>,
        prev: Option<BufferHandle>,
    ) -> Result<BufferHandle, MemoryError> {
        self.renderer.create_buffer_ext(descriptor, prev)
    }
}

impl<'a> Frame<'a> {
    pub fn begin_render_pass(
        &'a self,
        mut buf: command::CommandBuffer,
        render_pass: &Handle<render_pass::RenderPass>,
        target: &Handle<render_target::RenderTarget>,
        extent: util::Extent2D,
        clear_values: &[vk::ClearValue],
    ) -> Result<render_pass::RenderPassEncoder<'a>, command::CommandError> {
        let render_pass = self
            .renderer
            .resources
            .render_passes
            .get(render_pass)
            .expect("No such render pass");
        let target = self
            .renderer
            .resources
            .render_targets
            .get(target)
            .expect("TODO: Return error here");
        buf.begin_render_pass(&render_pass.0, &target.inner, extent, clear_values);

        Ok(render_pass::RenderPassEncoder::new(
            &self.renderer.resources,
            buf,
            self.renderer.frame_idx,
        ))
    }

    pub fn begin_presentation_pass(
        &'a self,
        buf: command::CommandBuffer,
        render_pass: &Handle<render_pass::RenderPass>,
    ) -> Result<render_pass::RenderPassEncoder<'a>, command::CommandError> {
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        self.begin_render_pass(
            buf,
            render_pass,
            self.renderer.current_present_target(),
            self.extent(),
            &clear_values,
        )
    }

    pub fn extent(&self) -> util::Extent2D {
        self.renderer.swapchain_extent()
    }

    pub fn finish(self) -> FinishedFrame {
        assert!(!self.recorded_command_buffers.is_empty());
        let Frame {
            recorded_command_buffers,
            gfx_command_pool,
            ..
        } = self;

        FinishedFrame {
            recorded_command_buffers,
            gfx_command_pool,
        }
    }
}

struct PresentationRenderTarget {
    render_pass: Handle<RenderPass>,
    swapchain_render_targets: Vec<Handle<render_target::RenderTarget>>,
    _depth_buffer: backend::image::ImageAttachment,
    _color_buffer: backend::image::ImageAttachment,
}

pub struct Renderer {
    resources: resource::Resources,

    // Swapchain-related
    presentation_render_target: Option<PresentationRenderTarget>,
    swapchain: swapchain::Swapchain,
    swapchain_image_idx: u32, // TODO: Bake this into the swapchain?
    image_to_frame_idx: Vec<Option<u32>>,

    loader: Option<Loader>,

    util_command_pool: command::CommandPool,

    // Needs to be kept-alive
    _debug_utils: backend::validation_layers::DebugUtils,

    frame_synchronization: [FrameSynchronization; MAX_FRAMES_IN_FLIGHT],
    frame_idx: u32,

    device: device::Device,
    surface: surface::Surface,
    instance: instance::Instance,
}

impl std::ops::Drop for Renderer {
    fn drop(&mut self) {
        // If we fail here, there is not much we can do, just log it.
        if let Err(e) = self.device.wait_idle() {
            log::error!("Failed to drop renderer: {}", e);
        }
    }
}

// Result holder struct
struct SwapchainAndCo {
    swapchain: swapchain::Swapchain,
    image_to_frame_idx: Vec<Option<u32>>,
}

fn create_swapchain_and_co(
    instance: &instance::Instance,
    device: &device::Device,
    surface: &surface::Surface,
    requested_extent: &util::Extent2D,
    old: Option<&swapchain::Swapchain>,
) -> Result<SwapchainAndCo, RenderError> {
    let swapchain = swapchain::Swapchain::new(instance, device, surface, requested_extent, old)?;

    let image_to_frame_idx: Vec<Option<u32>> = (0..swapchain.num_images()).map(|_| None).collect();
    Ok(SwapchainAndCo {
        swapchain,
        image_to_frame_idx,
    })
}

// Command-buffer related
impl Renderer {
    // TODO: Custom error?
    fn submit_blocking<F, R>(&self, f: F) -> Result<R, RenderError>
    where
        F: FnOnce(&mut command::CommandBuffer) -> R,
    {
        let mut command_buffer = self.util_command_pool.begin_single_submit()?;
        let r = f(&mut command_buffer);
        let fence = self.submit_command_buffer(command_buffer);
        fence.blocking_wait()?;
        Ok(r)
    }

    fn submit_command_buffer(&self, mut cmd_buffer: command::CommandBuffer) -> sync::Fence {
        cmd_buffer.end().expect("Failed to end command buffer");
        let done = sync::Fence::unsignaled(&self.device).expect("Failed to create fence");
        let buffers = [*cmd_buffer.vk_command_buffer()];
        let info = vk::SubmitInfo::builder().command_buffers(&buffers);

        self.device
            .graphics_queue()
            .submit(&info, &done)
            .expect("Failed to submit");

        done
    }
}

// Resource-related
impl Renderer {
    fn create_presentation_render_target(
        &mut self,
        format: util::Format,
        render_pass_h: Handle<RenderPass>,
    ) -> Result<PresentationRenderTarget, RenderError> {
        let render_pass = self
            .resources
            .render_passes
            .get(&render_pass_h)
            .expect("No presentation pass handle");
        let msaa_sample_count = render_pass.0.msaa_sample_count();
        let extent = self.swapchain_extent();
        let mip_levels = 1; // No mip maps
        let mem_usage = vma::MemoryUsage::AutoPreferDevice;

        let depth_buffer = {
            let format = self.device.depth_buffer_format().into();
            let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            let image = Image::empty_2d(
                &self.device.allocator(),
                ImageDescriptor {
                    extent,
                    format,
                    image_usage: usage,
                    image_flags: vk::ImageCreateFlags::empty(),
                    mem_usage,
                    mip_levels,
                    sample_count: msaa_sample_count,
                    array_layers: 1,
                },
            )
            .map_err(RenderError::RenderTargetImage)?;
            let image_view = backend::image::ImageView::new(
                &self.device,
                image.vk_image(),
                format,
                vk::ImageAspectFlags::DEPTH,
                mip_levels,
                vk::ImageViewType::TYPE_2D,
                0,
                1,
            )
            .map_err(RenderError::RenderTargetImageView)?;
            backend::image::ImageAttachment { image, image_view }
        };
        let color_buffer = {
            let usage =
                vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT;
            let mip_levels = 1; // No mip maps
            let image = Image::empty_2d(
                &self.device.allocator(),
                ImageDescriptor {
                    extent,
                    format,
                    image_usage: usage,
                    image_flags: vk::ImageCreateFlags::empty(),
                    mem_usage,
                    mip_levels,
                    sample_count: msaa_sample_count,
                    array_layers: 1,
                },
            )
            .map_err(RenderError::RenderTargetImage)?;

            let image_view = crate::backend::image::ImageView::new(
                &self.device,
                image.vk_image(),
                format,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
                vk::ImageViewType::TYPE_2D,
                0,
                1,
            )
            .map_err(RenderError::RenderTargetImageView)?;
            backend::image::ImageAttachment { image, image_view }
        };

        let mut swapchain_render_targets = Vec::with_capacity(self.swapchain.num_images());
        for image_view in self.swapchain.image_views() {
            let views = [
                &color_buffer.image_view,
                &depth_buffer.image_view,
                image_view,
            ];
            let fb = framebuffer::Framebuffer::new(
                &self.device,
                &views,
                &render_pass.0,
                &self.swapchain_extent(),
            )?;
            let handle = self
                .resources
                .render_targets
                .add(RenderTarget { inner: fb });
            swapchain_render_targets.push(handle);
        }

        Ok(PresentationRenderTarget {
            render_pass: render_pass_h,
            _color_buffer: color_buffer,
            _depth_buffer: depth_buffer,
            swapchain_render_targets,
        })
    }
}

impl Renderer {
    pub fn new<W>(window: &W, window_extent: util::Extent2D) -> Result<Self, RenderError>
    where
        W: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
    {
        let instance = instance::Instance::new(window)?;
        let _debug_utils = backend::validation_layers::DebugUtils::new(&instance)?;
        let surface = surface::Surface::new(&instance, window)?;
        let mut device = device::Device::new(&instance, &surface)?;

        let SwapchainAndCo {
            swapchain,
            image_to_frame_idx,
        } = create_swapchain_and_co(&instance, &device, &surface, &window_extent, None)?;

        let frame_synchronization = [
            FrameSynchronization::new(&device)?,
            FrameSynchronization::new(&device)?,
        ];

        let util_command_pool =
            command::CommandPool::new(&device, device.graphics_queue_family().clone())?;
        let descriptor_sets = pipeline_resource::PipelineResourceSetStorage::new(&device)?;
        let resources = resource::Resources {
            buffers: buffer::Buffers::default(),
            textures: texture::Textures::default(),
            graphics_pipelines: pipeline::GraphicsPipelines::default(),
            descriptor_sets,
            render_passes: resurs::Storage::default(),
            render_targets: resurs::Storage::default(),
        };

        let loader = Some(Loader::new(&mut device));
        let presentation_render_target = None;

        Ok(Self {
            instance,
            surface,
            device,
            swapchain,
            image_to_frame_idx,
            presentation_render_target,
            frame_synchronization,
            frame_idx: 0,
            swapchain_image_idx: 0,
            _debug_utils,
            resources,
            util_command_pool,
            loader,
        })
    }

    #[profiling::function]
    pub fn next_frame<'a, 'b: 'a>(&'b mut self) -> Result<Frame<'a>, RenderError> {
        {
            profiling::scope!("wait_and_acquire");
            let frame_sync = &mut self.frame_synchronization[self.frame_idx as usize];
            frame_sync.in_flight.blocking_wait()?;

            self.swapchain_image_idx = self
                .swapchain
                .acquire_next_image(Some(&frame_sync.image_available))?;
        }

        // This means that we received an image that might be in the process of rendering
        if let Some(mapped_frame_idx) = self.image_to_frame_idx[self.swapchain_image_idx as usize] {
            profiling::scope!("wait_image_in_use");
            self.frame_synchronization[mapped_frame_idx as usize]
                .in_flight
                .blocking_wait()?;
        }

        let frame_sync = &mut self.frame_synchronization[self.frame_idx as usize];
        let _ = frame_sync.command_pool.take();
        let gfx_command_pool =
            command::CommandPool::new(&self.device, self.device.graphics_queue_family().clone())?;

        self.image_to_frame_idx[self.swapchain_image_idx as usize] = Some(self.frame_idx);

        Ok(Frame::<'a> {
            renderer: self,
            recorded_command_buffers: Vec::new(),
            gfx_command_pool,
        })
    }

    #[profiling::function]
    pub fn submit(&mut self, frame: FinishedFrame) -> Result<(), RenderError> {
        assert!(
            !frame.recorded_command_buffers.is_empty(),
            "Needs atleast an empty render pass"
        );
        let frame_sync = &mut self.frame_synchronization[self.frame_idx as usize];
        assert!(
            frame_sync.command_pool.is_none(),
            "There should be no in-flight command buffers for this frame"
        );
        // It's fine to "drop" the command buffers, they will be all be destroyed/recycled with reset() when they are done
        // Their handles are passed submit info and stored in the driver before we deallocate the Vec holding them.
        let FinishedFrame {
            gfx_command_pool,
            recorded_command_buffers,
        } = frame;

        frame_sync.command_pool = Some(gfx_command_pool);

        let vk_wait_sems = [*frame_sync.image_available.vk_semaphore()];
        let wait_dst_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let vk_sig_sems = [*frame_sync.render_done.vk_semaphore()];

        let info = vk::SubmitInfo::builder()
            .wait_semaphores(&vk_wait_sems)
            .wait_dst_stage_mask(&wait_dst_mask)
            .signal_semaphores(&vk_sig_sems)
            .command_buffers(&recorded_command_buffers);

        let gfx_queue = self.device.graphics_queue();
        frame_sync.in_flight.reset()?;

        gfx_queue.submit(&info, &frame_sync.in_flight)?;

        let swapchains = [*self.swapchain.vk_swapchain()];
        let indices = [self.swapchain_image_idx];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&vk_sig_sems)
            .swapchains(&swapchains)
            .image_indices(&indices);

        let status = self
            .swapchain
            .enqueue_present(self.device.present_queue(), present_info.build())?;

        if let swapchain::SwapchainStatus::SubOptimal = status {
            return Err(RenderError::NeedsResize(ResizeReason::SubOptimal));
        }

        self.frame_idx = (self.frame_idx + 1) % MAX_FRAMES_IN_FLIGHT as u32;

        Ok(())
    }

    #[profiling::function]
    pub fn resize(&mut self, new_extent: util::Extent2D) -> Result<(), RenderError> {
        log::trace!(
            "Resizing from {} to {}",
            self.swapchain_extent(),
            new_extent
        );
        self.device.wait_idle()?;

        let SwapchainAndCo {
            swapchain,
            image_to_frame_idx,
        } = create_swapchain_and_co(
            &self.instance,
            &self.device,
            &self.surface,
            &new_extent,
            Some(&self.swapchain),
        )?;

        self.swapchain = swapchain;
        self.image_to_frame_idx = image_to_frame_idx;

        if let Some(prt) = self.presentation_render_target.take() {
            let rt = self.create_presentation_render_target(
                util::Format::from(self.swapchain.info().format),
                prt.render_pass,
            )?;
            self.presentation_render_target = Some(rt)
        }

        Ok(())
    }

    pub fn aspect_ratio(&self) -> f32 {
        let util::Extent2D { width, height } = self.swapchain_extent();

        width as f32 / height as f32
    }

    pub fn swapchain_extent(&self) -> util::Extent2D {
        self.swapchain.info().extent
    }

    pub fn loader(&mut self) -> Option<Loader> {
        self.loader.take()
    }

    pub fn resources()
}

/// Vulkan-specific
impl Renderer {
    pub fn presentation_render_pass(
        &mut self,
        msaa_sample_count: u8,
    ) -> Result<Handle<RenderPass>, RenderError> {
        let format = util::Format::from(self.swapchain.info().format);
        let render_pass =
            RenderPass::presentation_render_pass(&self.device, format, msaa_sample_count)?;
        let render_pass = self.resources.render_passes.add(render_pass);
        self.presentation_render_target =
            Some(self.create_presentation_render_target(format, render_pass)?);

        Ok(render_pass)
    }

    pub fn create_render_pass(
        &mut self,
        create_info: &vk::RenderPassCreateInfo,
    ) -> Result<Handle<RenderPass>, RenderError> {
        let rp = RenderPass::new_vk(&self.device, create_info)?;
        Ok(self.resources.render_passes.add(rp))
    }
}

/// These are functions only used by other parts of this lib
impl Renderer {
    fn current_present_target(&self) -> &Handle<render_target::RenderTarget> {
        &self
            .presentation_render_target
            .as_ref()
            .expect("Framebuffer has to have been created here")
            .swapchain_render_targets[self.swapchain_image_idx as usize]
    }

    pub(crate) fn resources_mut(&mut self) -> &'_ mut resource::Resources {
        &mut self.resources
    }
}

// TODO: Everything in this impl needs to be refactored
impl Renderer {
    fn update_descriptor_sets(&self, writes: &[vk::WriteDescriptorSet]) {
        unsafe {
            self.device.vk_device().update_descriptor_sets(writes, &[]);
        }
    }

    fn allocate_descriptor_sets(
        &mut self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> (
        Handle<pipeline_resource::PipelineResourceSet>,
        &[pipeline_resource::PipelineResourceSet; 2],
    ) {
        self.resources
            .descriptor_sets
            .alloc(bindings)
            .expect("Failed to alloc")
    }
}

impl Renderer {
    // TODO: Should this ensure that this is only called between frames?
    // This is only relevant for mutable buffers though so this should be fine?
    pub fn get_buffer(&self, handle: BufferHandle) -> Option<&DeviceBuffer> {
        self.resources.buffers.get(handle, self.frame_idx as usize)
    }

    /// Write the elements in `data` to the buffer in `h`, starting at element at `start`.
    /// This function handles alignment requirements properly.
    /// TODO: Accept frame to ensure that we can write to this buffer?
    pub fn write_buffer_at<T: bytemuck::Pod>(
        &mut self,
        h: BufferHandle,
        start: usize,
        data: &[T],
    ) -> Result<(), RenderError> {
        assert!(h.len() as usize >= data.len());
        assert_eq!(
            h.mutability(),
            BufferMutability::Mutable,
            "Can't modify immutable buffer"
        );

        let buf = self
            .resources
            .buffers
            .get_buffered_mut(h, self.frame_idx as usize)
            .ok_or_else(|| RenderError::InvalidHandle(h.handle().id()))?;

        let dst_offset = (h.offset() as usize + start) * buf.stride() as usize;
        let raw_data = util::as_byte_slice(data);
        buf.write(dst_offset, raw_data);
        Ok(())
    }

    pub fn write_buffer<T: bytemuck::Pod>(
        &mut self,
        h: BufferHandle,
        data: &[T],
    ) -> Result<(), RenderError> {
        self.write_buffer_at(h, 0, data)
    }

    // idx is in terms of T
    pub fn write_buffer_element<T: bytemuck::Pod>(
        &mut self,
        h: BufferHandle,
        data: &T,
        idx: usize,
    ) -> Result<(), RenderError> {
        self.write_buffer_at(h, idx, std::slice::from_ref(data))
    }

    pub fn create_buffer(
        &mut self,
        descriptor: BufferDescriptor<'_>,
    ) -> Result<BufferHandle, MemoryError> {
        // The transients that contain the data to upload need to outlive the wait for the command buffer to finish.
        // TOOD: Should descriptor.enqueue be unsafe?
        let (buf0, buf1) = self
            .submit_blocking(|cmd_buf| descriptor.enqueue(&self.device.allocator(), cmd_buf))
            .expect("Failed to submit commands to create buffer")
            .expect("Buffer creation failed");

        let handle = {
            let inner_handle = self
                .resources
                .buffers
                .add(buf0.buffer, buf1.map(|x| x.buffer));
            unsafe {
                BufferHandle::from_buffer(
                    inner_handle,
                    0,
                    descriptor.n_elems(),
                    descriptor.mutability(),
                    descriptor.buffer_type().ty(),
                )
            }
        };

        Ok(handle)
    }

    /// Recreates a whole buffer, regardless if the buffer handle is only a subslice
    /// Any handles pointing to this are invalidated and only the returned value should be
    /// used. If the buffer is not "Mutable" & Available, then this will panic.
    pub fn create_buffer_with(
        &mut self,
        descriptor: BufferDescriptor<'_>,
        prev: BufferHandle,
    ) -> Result<BufferHandle, MemoryError> {
        assert_eq!(prev.mutability(), BufferMutability::Mutable);
        // TODO: Check that self.frame_idx is not in-flight
        if let Some(ref mut buf) = self
            .resources
            .buffers
            .get_buffered_mut(prev, self.frame_idx as usize)
        {
            // TODO: Increment generation of backing storage
            buf.recreate(&self.device.allocator(), &descriptor)?;
            let handle = unsafe {
                BufferHandle::from_buffer(
                    *prev.handle(),
                    0,
                    descriptor.n_elems(),
                    BufferMutability::Mutable,
                    prev.buffer_type_id(),
                )
            };

            Ok(handle)
        } else {
            // TODO: Should this be a recoverable error?
            panic!("There is no buffer for: {prev:?}");
        }
    }

    /// TODO: Docs
    pub fn create_buffer_ext(
        &mut self,
        descriptor: BufferDescriptor<'_>,
        prev: Option<BufferHandle>,
    ) -> Result<BufferHandle, MemoryError> {
        if let Some(prev) = prev {
            self.create_buffer_with(descriptor, prev)
        } else {
            self.create_buffer(descriptor)
        }
    }
}

impl Renderer {
    pub fn get_pipeline(&self, handle: &Handle<GraphicsPipeline>) -> Option<&GraphicsPipeline> {
        self.resources.graphics_pipelines.get(handle)
    }

    pub fn create_gfx_pipeline(
        &mut self,
        descriptor: GraphicsPipelineDescriptor,
        render_pass: &Handle<RenderPass>,
    ) -> Result<Handle<GraphicsPipeline>, PipelineError> {
        let device = &self.device;
        let render_pass = self
            .resources
            .render_passes
            .get(render_pass)
            .expect("No such pass");
        let handle = self
            .resources
            .graphics_pipelines
            .get_or_add(descriptor, |d| d.create(device, &render_pass.0))?;

        Ok(handle)
    }
}

impl Renderer {
    pub fn get_texture(&self, handle: &Handle<Texture>) -> Option<&Texture> {
        self.resources.textures.get(handle)
    }

    pub fn create_texture(
        &mut self,
        descriptor: TextureDescriptor,
    ) -> Result<Handle<Texture>, TextureError> {
        let (desc, mipmaps, data) = descriptor.split_desc_data()?;
        let t = if let Some(data) = data {
            let (texture, _buffer) = self
                .submit_blocking(|command_buffer| {
                    texture::load_texture_from_data(
                        &self.device,
                        &self.device.allocator(),
                        command_buffer,
                        desc,
                        data.data(),
                        mipmaps,
                    )
                })
                .expect("TODO FAIL")
                .expect("TODO FAIL");
            texture
        } else {
            Texture::empty(&self.device, &self.device.allocator(), desc)?
        };

        Ok(self.resources_mut().textures.add(t))
    }

    pub fn generate_mipmaps(
        &mut self,
        handles: &[Handle<texture::Texture>],
    ) -> Result<(), RenderError> {
        use backend::vk::image::{generate_mipmaps, transition_image_layout};
        if handles.is_empty() {
            return Ok(());
        }
        let mut cmd_buf = self.util_command_pool.begin_single_submit()?;

        // TODO(perf): The old images need to be kept alive until we have submitted and waited for the cmd buffer
        // Try to use a free queue here instead. Use ManuallyDrop?
        let mut old_textures = Vec::new();
        for handle in handles {
            let texture = self
                .resources
                .textures
                .get_mut(handle)
                .ok_or_else(|| RenderError::InvalidHandle(handle.id()))?;
            log::trace!("Generating mipmap for {:p}", texture.vk_image());
            let extent = texture.extent();
            let format = texture.format();
            let mip_levels = texture::mip_levels_for(extent);
            let usage = vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT;
            let dst_image = Image::empty_2d(
                &self.device.allocator(),
                ImageDescriptor {
                    extent,
                    format,
                    image_usage: usage,
                    image_flags: vk::ImageCreateFlags::empty(),
                    mem_usage: vma::MemoryUsage::AutoPreferDevice,
                    mip_levels,
                    sample_count: vk::SampleCountFlags::TYPE_1,
                    array_layers: 1,
                },
            )
            .expect("Failed to create dst image");
            transition_image_layout(
                &mut cmd_buf,
                texture.vk_image(),
                1,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            transition_image_layout(
                &mut cmd_buf,
                dst_image.vk_image(),
                mip_levels,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            cmd_buf.copy_image(
                texture.vk_image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image.vk_image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                extent,
            );
            generate_mipmaps(&mut cmd_buf, &dst_image, extent, mip_levels);
            let new = texture::Texture::from_image(
                &self.device,
                dst_image,
                texture.sampler().descriptor(),
            )
            .expect("Failed to create mipmapped texture");
            old_textures.push(std::mem::replace(texture, new));
        }

        let done = self.submit_command_buffer(cmd_buf);
        done.blocking_wait().expect("Failed to wait for mipmapping");
        Ok(())
    }
}

impl Renderer {
    pub fn get_render_pass(&self, h: Handle<RenderPass>) -> Option<&RenderPass> {
        self.resources.render_passes.get(&h)
    }
}

impl Renderer {
    pub fn create_render_target(
        &mut self,
        render_pass: Handle<RenderPass>,
        attachments: &[&Handle<Texture>],
    ) -> Result<Handle<RenderTarget>, RenderError> {
        let render_pass = self
            .resources
            .render_passes
            .get(&render_pass)
            .ok_or_else(|| RenderError::InvalidHandle(render_pass.id()))?;
        let attachments: Result<Vec<&Texture>, RenderError> = attachments
            .iter()
            .map(|h| {
                self.resources
                    .textures
                    .get(h)
                    .ok_or_else(|| RenderError::InvalidHandle(h.id()))
            })
            .collect();
        let attachments = attachments?;
        let extent = attachments[0].extent();
        let data = RenderTarget::new(&self.device, &attachments, render_pass, &extent)?;
        let render_target = self.resources.render_targets.add(data);
        Ok(render_target)
    }

    // TODO: Refactor this. It seems a bit to narrow of a use-case to expose in the API here.
    pub fn create_cube_render_targets(
        &mut self,
        render_pass: Handle<RenderPass>,
        cube_map: Handle<Texture>,
    ) -> Result<[Handle<RenderTarget>; 6], RenderError> {
        let render_pass = self
            .resources
            .render_passes
            .get(&render_pass)
            .ok_or_else(|| RenderError::InvalidHandle(render_pass.id()))?;
        let texture = self
            .resources
            .textures
            .get(&cube_map)
            .ok_or_else(|| RenderError::InvalidHandle(cube_map.id()))?;

        let render_targets = std::array::from_fn(|idx| {
            let image_view = texture.sub_image_view(idx);
            let data =
                RenderTarget::new_raw(&self.device, &[image_view], render_pass, &texture.extent())
                    .expect("Failed to create render target");
            self.resources.render_targets.add(data)
        });

        Ok(render_targets)
    }
}
