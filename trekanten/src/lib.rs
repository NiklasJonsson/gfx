use ash::vk;

mod backend;
mod common;
pub mod descriptor;
mod error;
pub mod loader;
pub mod mem;
pub mod pipeline;
mod render_pass;
pub mod resource;
pub mod texture;
pub mod util;
pub mod vertex;

pub use error::RenderError;
pub use error::ResizeReason;
pub use loader::Loader;
pub use mem::{BufferHandle, BufferMutability};
pub use render_pass::RenderPassEncoder;
pub use resource::{Async, Handle, MutResourceManager, ResourceManager};

use ash::version::DeviceV1_0;
use backend::*;
use common::MAX_FRAMES_IN_FLIGHT;
use device::HasVkDevice;

use crate::mem::BufferDescriptor as _;

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
    pub fn new_raw_command_buffer(&self) -> Result<command::CommandBuffer, command::CommandError> {
        self.gfx_command_pool
            .create_command_buffer(command::CommandBufferSubmission::Single)
    }

    pub fn add_raw_command_buffer(&mut self, cmd_buffer: command::CommandBuffer) {
        self.recorded_command_buffers
            .push(*cmd_buffer.vk_command_buffer());
    }

    // TODO: Could we use vkCmdUpdateBuffer instead? Note that it can't be inside a render pass
    pub fn update_uniform_blocking<T: Copy>(
        &mut self,
        h: &BufferHandle<mem::UniformBuffer>,
        data: &T,
    ) -> Result<(), RenderError> {
        self.renderer.update_uniform(h, data)
    }

    pub fn begin_render_pass(
        &'a self,
    ) -> Result<render_pass::RenderPassEncoder<'a>, command::CommandError> {
        let mut buf = self.new_raw_command_buffer()?;
        buf.begin_render_pass(
            self.renderer.render_pass(),
            self.renderer.framebuffer(),
            self.renderer.swapchain_extent(),
        );

        Ok(render_pass::RenderPassEncoder::new(
            &self.renderer.resources,
            buf,
            self.renderer.frame_idx,
        ))
    }

    pub fn render_pass(&self) -> &backend::render_pass::RenderPass {
        self.renderer.render_pass()
    }

    pub fn extent(&self) -> util::Extent2D {
        self.renderer.swapchain_extent()
    }

    pub fn framebuffer(&self, _frame: &Frame) -> &framebuffer::Framebuffer {
        self.renderer.framebuffer()
    }

    pub fn finish(mut self) -> FinishedFrame {
        if self.recorded_command_buffers.is_empty() {
            // Add something just to clear the screen
            let buf = self
                .begin_render_pass()
                .expect("Failed to begin render pass")
                .build()
                .expect("Failed to build render command buffer");
            self.add_raw_command_buffer(buf);
        }

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

macro_rules! impl_mut_buffer_manager_frame {
    ($desc:ty, $resource:ty, $storage:ident) => {
        impl<'a> resource::MutResourceManager<$desc, $resource, BufferHandle<$resource>>
            for Frame<'a>
        {
            type Error = mem::MemoryError;

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

impl_mut_buffer_manager_frame!(
    mem::OwningVertexBufferDescriptor,
    mem::VertexBuffer,
    vertex_buffers
);

impl_mut_buffer_manager_frame!(
    mem::OwningIndexBufferDescriptor,
    mem::IndexBuffer,
    index_buffers
);

macro_rules! impl_buffer_manager_frame {
    ($desc:ty, $resource:ty, $handle:ty, $cmd_enum:ident, $storage:ident) => {
        impl<'a> resource::ResourceManager<$desc, $resource, $handle> for Frame<'a> {
            type Error = mem::MemoryError;

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
    mem::OwningVertexBufferDescriptor,
    mem::VertexBuffer,
    BufferHandle<mem::VertexBuffer>,
    CreateVertexBuffer,
    vertex_buffers
);

impl_buffer_manager_frame!(
    mem::OwningIndexBufferDescriptor,
    mem::IndexBuffer,
    BufferHandle<mem::IndexBuffer>,
    CreateIndexBuffer,
    index_buffers
);

// TODO: std::borrow::Cow here?
pub enum SyncResourceCommand {
    CreateVertexBuffer {
        descriptor: mem::OwningVertexBufferDescriptor,
    },
    CreateIndexBuffer {
        descriptor: mem::OwningIndexBufferDescriptor,
    },
    CreateUniformBuffer {
        descriptor: mem::OwningUniformBufferDescriptor,
    },
    CreateTexture {
        descriptor: texture::TextureDescriptor,
    },
    CreatePipeline {
        descriptor: pipeline::GraphicsPipelineDescriptor,
    },
}

pub enum PendingSyncResourceCommand {
    CreateVertexBuffer {
        handle: mem::BufferHandle<mem::VertexBuffer>,
        transients: [Option<mem::DeviceBuffer>; 2],
    },
    CreateIndexBuffer {
        handle: mem::BufferHandle<mem::IndexBuffer>,
        transients: [Option<mem::DeviceBuffer>; 2],
    },
    CreateUniformBuffer {
        handle: mem::BufferHandle<mem::UniformBuffer>,
        transients: [Option<mem::DeviceBuffer>; 2],
    },
    CreateTexture {
        handle: resurs::Handle<texture::Texture>,
        transients: mem::DeviceBuffer,
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
        handle: mem::BufferHandle<mem::VertexBuffer>,
    },
    CreateIndexBuffer {
        handle: mem::BufferHandle<mem::IndexBuffer>,
    },
    CreateUniformBuffer {
        handle: mem::BufferHandle<mem::UniformBuffer>,
    },
    CreateTexture {
        handle: resurs::Handle<texture::Texture>,
    },
    CreatePipeline {
        handle: resurs::Handle<pipeline::GraphicsPipeline>,
    },
}

pub struct Renderer {
    resources: resource::Resources,

    // Swapchain-related
    // TODO: render pass should move to something like a render graph
    render_pass: backend::render_pass::RenderPass,
    swapchain_framebuffers: Vec<framebuffer::Framebuffer>,
    depth_buffer: depth_buffer::DepthBuffer,
    color_buffer: color_buffer::ColorBuffer,
    swapchain: swapchain::Swapchain,
    swapchain_image_idx: u32, // TODO: Bake this into the swapchain?
    image_to_frame_idx: Vec<Option<u32>>,

    loader: Option<Loader>,

    util_command_pool: command::CommandPool,

    // Needs to be kept-alive
    _debug_utils: util::vk_debug::DebugUtils,

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
    depth_buffer: depth_buffer::DepthBuffer,
    color_buffer: color_buffer::ColorBuffer,
    swapchain_framebuffers: Vec<framebuffer::Framebuffer>,
    image_to_frame_idx: Vec<Option<u32>>,
    render_pass: backend::render_pass::RenderPass,
}

fn create_swapchain_and_co(
    instance: &instance::Instance,
    device: &device::Device,
    surface: &surface::Surface,
    requested_extent: &util::Extent2D,
    old: Option<&swapchain::Swapchain>,
) -> Result<SwapchainAndCo, RenderError> {
    let msaa_sample_count = device.max_msaa_sample_count();
    let swapchain =
        swapchain::Swapchain::new(&instance, &device, &surface, &requested_extent, old)?;
    let render_pass =
        backend::render_pass::RenderPass::new(&device, swapchain.info().format, msaa_sample_count)?;

    let image_to_frame_idx: Vec<Option<u32>> = (0..swapchain.num_images()).map(|_| None).collect();
    let depth_buffer =
        depth_buffer::DepthBuffer::new(device, &swapchain.info().extent, msaa_sample_count)?;
    let color_buffer = color_buffer::ColorBuffer::new(
        device,
        swapchain.info().format.into(),
        &swapchain.info().extent,
        msaa_sample_count,
    )?;
    let swapchain_framebuffers =
        swapchain.create_framebuffers_for(&render_pass, &depth_buffer, &color_buffer)?;

    Ok(SwapchainAndCo {
        swapchain,
        depth_buffer,
        color_buffer,
        swapchain_framebuffers,
        image_to_frame_idx,
        render_pass,
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

                let handle = self.resources.textures.add(&descriptor, image);
                Some(PendingSyncResourceCommand::CreateTexture { handle, transients })
            }
            _ => unreachable!(),
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
    ) -> Result<FinishedResourceCommand, mem::MemoryError> {
        if let SyncResourceCommand::CreatePipeline { descriptor } = command {
            let render_pass = &self.render_pass;
            let device = &self.device;
            let handle = self
                .resources
                .graphics_pipelines
                .get_or_add(descriptor, |d| d.create(device, render_pass))
                .unwrap();
            Ok(FinishedResourceCommand::CreatePipeline { handle })
        } else {
            let mut raw_cmd_buf = self.util_command_pool.begin_single_submit()?;
            let pending_cmd = self
                .schedule_command(command, &mut raw_cmd_buf)
                .expect("Should be pending");

            let done = self.submit_command_buffer(raw_cmd_buf);
            done.blocking_wait()
                .expect("Failed to wait for resource creation");
            Ok(pending_cmd.done())
        }
    }
}

impl Renderer {
    pub fn new<W>(window: &W, window_extent: util::Extent2D) -> Result<Self, RenderError>
    where
        W: raw_window_handle::HasRawWindowHandle,
    {
        let instance = instance::Instance::new(window)?;
        let _debug_utils = util::vk_debug::DebugUtils::new(&instance)?;
        let surface = surface::Surface::new(&instance, window)?;
        let mut device = device::Device::new(&instance, &surface)?;

        let SwapchainAndCo {
            swapchain,
            swapchain_framebuffers,
            depth_buffer,
            color_buffer,
            image_to_frame_idx,
            render_pass,
        } = create_swapchain_and_co(&instance, &device, &surface, &window_extent, None)?;

        let frame_synchronization = [
            FrameSynchronization::new(&device)?,
            FrameSynchronization::new(&device)?,
        ];

        let util_command_pool =
            command::CommandPool::new(&device, device.graphics_queue_family().clone())?;
        let descriptor_sets = descriptor::DescriptorSets::new(&device)?;
        let resources = resource::Resources {
            uniform_buffers: mem::UniformBuffers::default(),
            vertex_buffers: mem::VertexBuffers::default(),
            index_buffers: mem::IndexBuffers::default(),
            textures: texture::Textures::default(),
            graphics_pipelines: pipeline::GraphicsPipelines::default(),
            descriptor_sets,
        };

        let loader = Some(Loader::new(&mut device));

        Ok(Self {
            instance,
            surface,
            device,
            swapchain,
            image_to_frame_idx,
            render_pass,
            swapchain_framebuffers,
            depth_buffer,
            color_buffer,
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
            let frame_sync = &mut self.frame_synchronization[self.frame_idx as usize];
            frame_sync.in_flight.blocking_wait()?;

            self.swapchain_image_idx = self
                .swapchain
                .acquire_next_image(Some(&frame_sync.image_available))?;
        }

        // This means that we received an image that might be in the process of rendering
        if let Some(mapped_frame_idx) = self.image_to_frame_idx[self.swapchain_image_idx as usize] {
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
            swapchain_framebuffers,
            depth_buffer,
            color_buffer,
            image_to_frame_idx,
            render_pass,
        } = create_swapchain_and_co(
            &self.instance,
            &self.device,
            &self.surface,
            &new_extent,
            Some(&self.swapchain),
        )?;

        self.swapchain = swapchain;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.depth_buffer = depth_buffer;
        self.color_buffer = color_buffer;
        self.image_to_frame_idx = image_to_frame_idx;
        self.render_pass = render_pass;

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

/// These are functions only used by other parts of this lib
impl Renderer {
    fn update_uniform<T: Copy>(
        &mut self,
        h: &BufferHandle<mem::UniformBuffer>,
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

    fn render_pass(&self) -> &backend::render_pass::RenderPass {
        &self.render_pass
    }

    fn framebuffer(&self) -> &framebuffer::Framebuffer {
        &self.swapchain_framebuffers[self.swapchain_image_idx as usize]
    }

    pub(crate) fn resources_mut(&mut self) -> &'_ mut resource::Resources {
        &mut self.resources
    }
}

// TODO: Everything in this impl needs to be refactored
impl Renderer {
    fn update_descriptor_sets(&self, writes: &[vk::WriteDescriptorSet]) {
        unsafe {
            self.device.vk_device().update_descriptor_sets(&writes, &[]);
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
        impl resource::ResourceManager<$desc, $resource, $handle> for Renderer {
            type Error = mem::MemoryError;

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
    mem::OwningVertexBufferDescriptor,
    mem::VertexBuffer,
    BufferHandle<mem::VertexBuffer>,
    CreateVertexBuffer,
    vertex_buffers
);
impl_buffer_manager!(
    mem::OwningIndexBufferDescriptor,
    mem::IndexBuffer,
    BufferHandle<mem::IndexBuffer>,
    CreateIndexBuffer,
    index_buffers
);
impl_buffer_manager!(
    mem::OwningUniformBufferDescriptor,
    mem::UniformBuffer,
    BufferHandle<mem::UniformBuffer>,
    CreateUniformBuffer,
    uniform_buffers
);

macro_rules! impl_resource_manager {
    ($desc:ty, $resource:ty, $handle:ty, $error:ty, $cmd_enum:ident, $storage:ident) => {
        impl resource::ResourceManager<$desc, $resource, $handle> for Renderer {
            type Error = $error;

            fn get_resource(&self, handle: &$handle) -> Option<&$resource> {
                self.resources.$storage.get(handle)
            }

            fn create_resource_blocking(
                &mut self,
                descriptor: $desc,
            ) -> Result<$handle, Self::Error> {
                if let Some(handle) = self.resources.$storage.cached(&descriptor) {
                    return Ok(handle);
                }

                let cmd = SyncResourceCommand::$cmd_enum { descriptor };
                let finished = self.execute_command(cmd).unwrap();
                match finished {
                    FinishedResourceCommand::$cmd_enum { handle } => Ok(handle),
                    _ => unreachable!(),
                }
            }
        }
    };
}
impl_resource_manager!(
    texture::TextureDescriptor,
    texture::Texture,
    Handle<texture::Texture>,
    texture::TextureError,
    CreateTexture,
    textures
);

impl_resource_manager!(
    pipeline::GraphicsPipelineDescriptor,
    pipeline::GraphicsPipeline,
    Handle<pipeline::GraphicsPipeline>,
    pipeline::PipelineError,
    CreatePipeline,
    graphics_pipelines
);
