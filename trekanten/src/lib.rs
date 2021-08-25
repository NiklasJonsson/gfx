use ash::vk;

pub use ash::vk as raw_vk;

mod backend;
pub mod buffer;
mod common;
pub mod descriptor;
mod error;
pub mod loader;
pub mod pipeline;
mod render_pass;
mod render_target;
pub mod resource;
pub mod texture;
pub mod traits;
pub mod util;
pub mod vertex;

pub use backend::command::CommandBuffer;
pub use buffer::{BufferHandle, BufferMutability};
pub use error::RenderError;
pub use error::ResizeReason;
pub use loader::Loader;
pub use render_pass::{RenderPass, RenderPassEncoder};
pub use render_target::RenderTarget;
pub use resource::{Async, Handle, MutResourceManager, ResourceManager};
pub use texture::Texture;
pub use traits::{PushConstant, Std140, Uniform};

pub use trekanten_derive::Std140Compat;

use ash::version::DeviceV1_0;
use backend::device::HasVkDevice as _;
use backend::{command, device, framebuffer, instance, surface, swapchain, sync};
use common::MAX_FRAMES_IN_FLIGHT;

use crate::backend::vk::{buffer::Buffer, image::Image, MemoryError};

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

    // TODO: Could we use vkCmdUpdateBuffer instead? Note that it can't be inside a render pass
    pub fn update_uniform_blocking<T: Copy + traits::Uniform>(
        &mut self,
        h: &BufferHandle<buffer::DeviceUniformBuffer>,
        data: &T,
    ) -> Result<(), RenderError> {
        self.renderer.update_uniform(h, data)
    }

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

// TODO: Refactor?
impl<'a> Frame<'a> {
    pub fn get_texture(&self, handle: &Handle<Texture>) -> Option<&Texture> {
        self.renderer.get_texture(handle)
    }
}

macro_rules! impl_mut_buffer_manager_frame {
    ($desc:ty, $resource:ty, $storage:ident) => {
        impl<'a, 'b> resource::MutResourceManager<$desc, $resource, BufferHandle<$resource>>
            for Frame<'a>
        {
            type Error = MemoryError;

            /// Recreates a whole buffer, regardless if the buffer handle is only a subslice
            /// Any handles pointing to this are invalidated and only the returned value should be
            /// used. If the buffer is not "Mutable" & Available, then this will panic.
            fn recreate_resource_blocking(
                &mut self,
                handle: BufferHandle<$resource>,
                descriptor: $desc,
            ) -> Result<BufferHandle<$resource>, Self::Error> {
                assert_eq!(handle.mutability(), BufferMutability::Mutable);
                if let Some(ref mut buf) = self
                    .renderer
                    .resources
                    .$storage
                    .get_buffered_mut(&handle, self.renderer.frame_idx as usize)
                {
                    // TODO: Increment generation of backing storage
                    buf.recreate(&self.renderer.device.allocator(), &descriptor)?;
                    let handle = unsafe {
                        BufferHandle::from_buffer(
                            *handle.handle(),
                            0,
                            descriptor.n_elems(),
                            BufferMutability::Mutable,
                        )
                    };

                    Ok(handle)
                } else {
                    panic!("Can't recreate a pending resource!");
                }
            }
        }
    };
}

// TODO: How to avoid having to give <'b> here?
impl_mut_buffer_manager_frame!(
    buffer::VertexBufferDescriptor<'b>,
    buffer::DeviceVertexBuffer,
    vertex_buffers
);

impl_mut_buffer_manager_frame!(
    buffer::IndexBufferDescriptor<'b>,
    buffer::DeviceIndexBuffer,
    index_buffers
);

macro_rules! impl_buffer_manager_frame {
    ($desc:ty, $resource:ty, $handle:ty, $cmd_enum:ident, $storage:ident) => {
        impl<'a, 'b> resource::ResourceManager<$desc, $resource, $handle> for Frame<'a> {
            type Error = MemoryError;

            fn get_resource(&self, handle: &$handle) -> Option<&$resource> {
                self.renderer.get_resource(handle)
            }

            fn create_resource_blocking(
                &mut self,
                descriptor: $desc,
            ) -> Result<$handle, Self::Error> {
                self.renderer.create_resource_blocking(descriptor)
            }
        }
    };
}

impl_buffer_manager_frame!(
    buffer::VertexBufferDescriptor<'b>,
    buffer::DeviceVertexBuffer,
    BufferHandle<buffer::DeviceVertexBuffer>,
    CreateVertexBuffer,
    vertex_buffers
);

impl_buffer_manager_frame!(
    buffer::IndexBufferDescriptor<'b>,
    buffer::DeviceIndexBuffer,
    BufferHandle<buffer::DeviceIndexBuffer>,
    CreateIndexBuffer,
    index_buffers
);

pub enum SyncResourceCommand<'a> {
    CreateVertexBuffer {
        descriptor: buffer::VertexBufferDescriptor<'a>,
    },
    CreateIndexBuffer {
        descriptor: buffer::IndexBufferDescriptor<'a>,
    },
    CreateUniformBuffer {
        descriptor: buffer::UniformBufferDescriptor<'a>,
    },
    CreateTexture {
        descriptor: texture::TextureDescriptor,
    },
}

pub enum PendingSyncResourceCommand {
    CreateVertexBuffer {
        handle: buffer::BufferHandle<buffer::DeviceVertexBuffer>,
        transients: [Option<Buffer>; 2],
    },
    CreateIndexBuffer {
        handle: buffer::BufferHandle<buffer::DeviceIndexBuffer>,
        transients: [Option<Buffer>; 2],
    },
    CreateUniformBuffer {
        handle: buffer::BufferHandle<buffer::DeviceUniformBuffer>,
        transients: [Option<Buffer>; 2],
    },
    CreateTexture {
        handle: resurs::Handle<texture::Texture>,
        transients: Buffer,
    },
}

impl PendingSyncResourceCommand {
    fn done(self) -> FinishedResourceCommand {
        match self {
            Self::CreateVertexBuffer { handle, .. } => {
                FinishedResourceCommand::CreateVertexBuffer { handle }
            }
            Self::CreateIndexBuffer { handle, .. } => {
                FinishedResourceCommand::CreateIndexBuffer { handle }
            }
            Self::CreateUniformBuffer { handle, .. } => {
                FinishedResourceCommand::CreateUniformBuffer { handle }
            }
            Self::CreateTexture { handle, .. } => FinishedResourceCommand::CreateTexture { handle },
        }
    }
}

pub enum FinishedResourceCommand {
    CreateVertexBuffer {
        handle: buffer::BufferHandle<buffer::DeviceVertexBuffer>,
    },
    CreateIndexBuffer {
        handle: buffer::BufferHandle<buffer::DeviceIndexBuffer>,
    },
    CreateUniformBuffer {
        handle: buffer::BufferHandle<buffer::DeviceUniformBuffer>,
    },
    CreateTexture {
        handle: resurs::Handle<texture::Texture>,
    },
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

macro_rules! process_buffer_creation {
    ($cmd:ident, $desc:ident, $self:ident, $cmd_buffer:ident, $storage:ident) => {{
        let (buf0, buf1) = $desc
            .enqueue(&$self.device.allocator(), $cmd_buffer)
            .expect("Fail");

        let (buffer1, transient1) = if let Some(buf1) = buf1 {
            (Some(buf1.buffer), buf1.transient)
        } else {
            (None, None)
        };

        let handle = {
            let inner_handle = $self.resources.$storage.add(buf0.buffer, buffer1);
            unsafe {
                BufferHandle::from_buffer(inner_handle, 0, $desc.n_elems(), $desc.mutability())
            }
        };

        let transients = [buf0.transient, transient1];
        Some(PendingSyncResourceCommand::$cmd { handle, transients })
    }};
}

// Resource-related
impl Renderer {
    fn schedule_command(
        &mut self,
        command: SyncResourceCommand,
        cmd_buffer: &mut command::CommandBuffer,
    ) -> Option<PendingSyncResourceCommand> {
        match command {
            SyncResourceCommand::CreateVertexBuffer { descriptor } => {
                process_buffer_creation!(
                    CreateVertexBuffer,
                    descriptor,
                    self,
                    cmd_buffer,
                    vertex_buffers
                )
            }
            SyncResourceCommand::CreateIndexBuffer { descriptor } => {
                process_buffer_creation!(
                    CreateIndexBuffer,
                    descriptor,
                    self,
                    cmd_buffer,
                    index_buffers
                )
            }
            SyncResourceCommand::CreateUniformBuffer { descriptor } => {
                process_buffer_creation!(
                    CreateUniformBuffer,
                    descriptor,
                    self,
                    cmd_buffer,
                    uniform_buffers
                )
            }
            SyncResourceCommand::CreateTexture { descriptor } => {
                let (image, transients) = descriptor
                    .enqueue(&self.device.allocator(), &self.device, cmd_buffer)
                    .expect("Fail");

                let handle = self.resources.textures.add(image);
                Some(PendingSyncResourceCommand::CreateTexture { handle, transients })
            }
        }
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

    fn execute_command(
        &mut self,
        command: SyncResourceCommand,
    ) -> Result<FinishedResourceCommand, MemoryError> {
        let mut raw_cmd_buf = self.util_command_pool.begin_single_submit()?;
        let pending_cmd = self
            .schedule_command(command, &mut raw_cmd_buf)
            .expect("Should be pending");

        let done = self.submit_command_buffer(raw_cmd_buf);
        done.blocking_wait()
            .expect("Failed to wait for resource creation");
        Ok(pending_cmd.done())
    }

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
        let props = vk_mem::MemoryUsage::GpuOnly;

        let depth_buffer = {
            let format = self.device.depth_buffer_format().into();
            let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            let image = Image::empty_2d(
                &self.device.allocator(),
                extent,
                format,
                usage,
                props,
                mip_levels,
                msaa_sample_count,
            )
            .map_err(RenderError::RenderTargetImage)?;
            let image_view = backend::image::ImageView::new(
                &self.device,
                image.vk_image(),
                format,
                vk::ImageAspectFlags::DEPTH,
                mip_levels,
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
                extent,
                format,
                usage,
                props,
                mip_levels,
                msaa_sample_count,
            )
            .map_err(RenderError::RenderTargetImage)?;

            let image_view = crate::backend::image::ImageView::new(
                &self.device,
                image.vk_image(),
                format,
                vk::ImageAspectFlags::COLOR,
                mip_levels,
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
        W: raw_window_handle::HasRawWindowHandle,
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
        let descriptor_sets = descriptor::DescriptorSets::new(&device)?;
        let resources = resource::Resources {
            uniform_buffers: buffer::UniformBuffers::default(),
            vertex_buffers: buffer::VertexBuffers::default(),
            index_buffers: buffer::IndexBuffers::default(),
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
        let _ = std::mem::replace(&mut frame_sync.command_pool, None);
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
    fn update_uniform<T: Copy + traits::Uniform>(
        &mut self,
        h: &BufferHandle<buffer::DeviceUniformBuffer>,
        data: &T,
    ) -> Result<(), RenderError> {
        let ubuf = self
            .resources
            .uniform_buffers
            .get_buffered_mut(h, self.frame_idx as usize)
            .ok_or_else(|| RenderError::InvalidHandle(h.handle().id()))?;

        ubuf.update_with(data, h.idx() as u64)
            .map_err(RenderError::UniformBuffer)
    }

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
        Handle<descriptor::DescriptorSet>,
        &[descriptor::DescriptorSet; 2],
    ) {
        self.resources
            .descriptor_sets
            .alloc(bindings)
            .expect("Failed to alloc")
    }
}

macro_rules! impl_buffer_manager {
    ($desc:ty, $resource:ty, $handle:ty, $cmd_enum:ident, $storage:ident) => {
        impl<'a> resource::ResourceManager<$desc, $resource, $handle> for Renderer {
            type Error = MemoryError;

            fn get_resource(&self, handle: &$handle) -> Option<&$resource> {
                self.resources.$storage.get(handle, self.frame_idx as usize)
            }

            fn create_resource_blocking(
                &mut self,
                descriptor: $desc,
            ) -> Result<$handle, Self::Error> {
                let cmd = SyncResourceCommand::$cmd_enum { descriptor };
                let finished = self.execute_command(cmd)?;
                match finished {
                    FinishedResourceCommand::$cmd_enum { handle } => Ok(handle),
                    _ => unreachable!(),
                }
            }
        }
    };
}

impl_buffer_manager!(
    buffer::VertexBufferDescriptor<'a>,
    buffer::DeviceVertexBuffer,
    BufferHandle<buffer::DeviceVertexBuffer>,
    CreateVertexBuffer,
    vertex_buffers
);
impl_buffer_manager!(
    buffer::IndexBufferDescriptor<'a>,
    buffer::DeviceIndexBuffer,
    BufferHandle<buffer::DeviceIndexBuffer>,
    CreateIndexBuffer,
    index_buffers
);
impl_buffer_manager!(
    buffer::UniformBufferDescriptor<'a>,
    buffer::DeviceUniformBuffer,
    BufferHandle<buffer::DeviceUniformBuffer>,
    CreateUniformBuffer,
    uniform_buffers
);

use pipeline::{GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError};

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

use crate::texture::{TextureDescriptor, TextureError};
impl Renderer {
    pub fn get_texture(&self, handle: &Handle<Texture>) -> Option<&Texture> {
        self.resources.textures.get(handle)
    }

    pub fn create_texture(
        &mut self,
        descriptor: TextureDescriptor,
    ) -> Result<Handle<Texture>, TextureError> {
        if !descriptor.needs_command_buffer() {
            let t = Texture::create_no_cmds(&self.device, &self.device.allocator(), &descriptor)?;
            return Ok(self.resources.textures.add(t));
        }

        let cmd = SyncResourceCommand::CreateTexture { descriptor };
        let finished = self.execute_command(cmd).unwrap();
        match finished {
            FinishedResourceCommand::CreateTexture { handle } => Ok(handle),
            _ => unreachable!(),
        }
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
            let extent = texture.extent();
            let format = texture.format();
            let mip_levels = texture::mip_levels_for(extent);
            let usage = vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED;
            let dst_image = Image::empty_2d(
                &self.device.allocator(),
                extent,
                format,
                usage,
                vk_mem::MemoryUsage::GpuOnly,
                mip_levels,
                vk::SampleCountFlags::TYPE_1,
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
                &extent,
            );
            generate_mipmaps(&mut cmd_buf, dst_image.vk_image(), &extent, mip_levels);
            let new =
                texture::Texture::from_device_image(&self.device, dst_image, format, mip_levels)
                    .expect("Failed to create mipmapped texture");
            old_textures.push(std::mem::replace(texture, new));
        }

        let done = self.submit_command_buffer(cmd_buf);
        done.blocking_wait().expect("Failed to wait for mipmapping");
        Ok(())
    }
}

impl Renderer {
    pub fn create_render_target(
        &mut self,
        render_pass: &Handle<RenderPass>,
        attachments: &[&Handle<Texture>],
    ) -> Result<Handle<RenderTarget>, RenderError> {
        let render_pass = self
            .resources
            .render_passes
            .get(render_pass)
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
}
