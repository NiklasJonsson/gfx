use ash::vk;

mod backend;
mod common;
pub mod descriptor;
mod error;
pub mod loader;
mod mem;
pub mod mesh;
pub mod pipeline;
mod render_pass;
pub mod resource;
pub mod texture;
pub mod uniform;
pub mod util;
pub mod vertex;

pub use error::RenderError;
pub use error::ResizeReason;
pub use mem::BufferHandle;
pub use resource::{Handle, MutResourceManager, ResourceManager};

use ash::version::DeviceV1_0;
use backend::*;
use common::MAX_FRAMES_IN_FLIGHT;
use device::HasVkDevice;
use resource::ResourceCommand;

use std::sync::Arc;

use crate::mem::BufferDescriptor as _;

// Notes:
// We can have N number of swapchain images, it depends on the backing presentation implementation.
// Generally, we are aiming for three images + MAILBOX (render one and use the latest of the two waiting)
//
// We use MAX_FRAMES_IN_FLIGHT (2, hardcoded atm) frames in flight at once. This allows us to start the next frame directly after we render.
// Whenever next_frame() is called, it can be thought of as binding one of the two frames to a particular swapchain image.
// All rendering in that frame will be done on that swapchain image/framebuffer.

pub struct FrameSynchronization {
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
        h: &BufferHandle<uniform::UniformBuffer>,
        data: &T,
    ) -> Result<(), RenderError> {
        self.renderer.update_uniform(h, data)
    }

    // DrawList
    pub fn begin_render_pass(
        &'a self,
    ) -> Result<render_pass::RenderPassBuilder<'a>, command::CommandError> {
        let mut buf = self.new_raw_command_buffer()?;
        buf.begin_render_pass(
            self.renderer.render_pass(),
            self.renderer.framebuffer(),
            self.renderer.swapchain_extent(),
        );

        Ok(render_pass::RenderPassBuilder::new(
            self.renderer,
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

struct Resources {
    descriptor_sets: descriptor::DescriptorSets,
}

impl Resources {
    fn new(descriptor_sets: descriptor::DescriptorSets) -> Self {
        Self { descriptor_sets }
    }
}

// The curse of typed buffers
enum PendingResourceCommand {
    CreateVertexBuffer {
        descriptor: mesh::OwningVertexBufferDescriptor,
        handle: mem::BufferHandle<mesh::VertexBuffer>,
        buffer0: mesh::VertexBuffer,
        buffer1: Option<mesh::VertexBuffer>, // For double buffering
        transients: [Option<mem::DeviceBuffer>; 2],
    },
    CreateIndexBuffer {
        descriptor: mesh::OwningIndexBufferDescriptor,
        handle: mem::BufferHandle<mesh::IndexBuffer>,
        buffer0: mesh::IndexBuffer,
        buffer1: Option<mesh::IndexBuffer>, // For double buffering
        transients: [Option<mem::DeviceBuffer>; 2],
    },
    CreateUniformBuffer {
        descriptor: uniform::OwningUniformBufferDescriptor,
        handle: mem::BufferHandle<uniform::UniformBuffer>,
        buffer0: uniform::UniformBuffer,
        buffer1: Option<uniform::UniformBuffer>, // For double buffering
        transients: [Option<mem::DeviceBuffer>; 2],
    },
    CreateTexture {
        descriptor: texture::TextureDescriptor,
        handle: resurs::Handle<texture::Texture>,
        image: texture::Texture,
        transients: mem::DeviceBuffer,
    },
}
struct PendingResourceJob {
    batch: Vec<PendingResourceCommand>,
    done: sync::Fence,
}

pub struct Renderer {
    resources: Resources,
    async_resources: Arc<resource::AsyncResources>,

    // Swapchain-related
    // TODO: render pass should move to something like a render graph
    render_pass: backend::render_pass::RenderPass,
    swapchain_framebuffers: Vec<framebuffer::Framebuffer>,
    depth_buffer: depth_buffer::DepthBuffer,
    color_buffer: color_buffer::ColorBuffer,
    swapchain: swapchain::Swapchain,
    swapchain_image_idx: u32, // TODO: Bake this into the swapchain?
    image_to_frame_idx: Vec<Option<u32>>,

    loader: loader::Loader,
    resource_cmd_receive_queue: loader::ResourceCommandReceiver,
    pending_resource_jobs: Vec<PendingResourceJob>,

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
    ($cmd:ident, $desc:ident, $self:ident, $cmd_buffer:ident, $handle:ident) => {{
        let (buf0, buf1) = $desc
            .enqueue(
                &$self.device,
                $cmd_buffer.expect("This needs a command buffer"),
            )
            .expect("Fail");

        let (buffer1, transient1) = if let Some(buf1) = buf1 {
            (Some(buf1.buffer), buf1.transient)
        } else {
            (None, None)
        };

        let buffer0 = buf0.buffer;
        let transients = [buf0.transient, transient1];
        Some(PendingResourceCommand::$cmd {
            descriptor: $desc,
            handle: $handle,
            buffer0,
            buffer1,
            transients,
        })
    }};
}

// Resource-related
impl Renderer {
    fn process_command(
        &self,
        command: ResourceCommand,
        cmd_buffer: Option<&mut command::CommandBuffer>,
    ) -> Option<PendingResourceCommand> {
        match command {
            ResourceCommand::CreateVertexBuffer { handle, descriptor } => {
                process_buffer_creation!(CreateVertexBuffer, descriptor, self, cmd_buffer, handle)
            }
            ResourceCommand::CreateIndexBuffer { handle, descriptor } => {
                process_buffer_creation!(CreateIndexBuffer, descriptor, self, cmd_buffer, handle)
            }
            ResourceCommand::CreateUniformBuffer { handle, descriptor } => {
                process_buffer_creation!(CreateUniformBuffer, descriptor, self, cmd_buffer, handle)
            }
            ResourceCommand::CreateTexture { handle, descriptor } => {
                let (image, transients) = descriptor
                    .enqueue(
                        &self.device,
                        cmd_buffer.expect("texture creation needs command buffer"),
                    )
                    .expect("Fail");
                Some(PendingResourceCommand::CreateTexture {
                    descriptor,
                    handle,
                    image,
                    transients,
                })
            }
            ResourceCommand::CreatePipeline { handle, descriptor } => {
                assert!(
                    cmd_buffer.is_none(),
                    "No need for a command buffer for pipeline creation"
                );
                self.async_resources.graphics_pipelines.insert(
                    &handle,
                    descriptor.create(&self.device, &self.render_pass).unwrap(),
                );
                None
            }
        }
    }

    fn finish_command(&self, command: PendingResourceCommand) {
        // TODO: Pass the transients back to a free list in the allocator
        match command {
            PendingResourceCommand::CreateVertexBuffer {
                handle,
                buffer0,
                buffer1,
                transients: _transients,
                descriptor: _descriptor,
            } => self
                .async_resources
                .vertex_buffers
                .insert(&handle, buffer0, buffer1),
            PendingResourceCommand::CreateIndexBuffer {
                handle,
                buffer0,
                buffer1,
                transients: _transients,
                descriptor: _descriptor,
            } => self
                .async_resources
                .index_buffers
                .insert(&handle, buffer0, buffer1),
            PendingResourceCommand::CreateUniformBuffer {
                handle,
                buffer0,
                buffer1,
                transients: _transients,
                descriptor: _descriptor,
            } => self
                .async_resources
                .uniform_buffers
                .insert(&handle, buffer0, buffer1),
            PendingResourceCommand::CreateTexture {
                handle,
                image,
                transients: _transients,
                descriptor: _descriptor,
            } => self.async_resources.textures.insert(&handle, image),
        }
    }

    fn submit_resource_job(
        &self,
        batch: Vec<PendingResourceCommand>,
        mut cmd_buffer: command::CommandBuffer,
    ) -> PendingResourceJob {
        cmd_buffer.end().expect("Failed to end command buffer");
        let done = sync::Fence::unsignaled(&self.device).expect("Failed to create fence");
        let buffers = [*cmd_buffer.vk_command_buffer()];
        let info = vk::SubmitInfo::builder().command_buffers(&buffers);

        self.device
            .util_queue()
            .submit(&info, &done)
            .expect("Failed to submit");

        PendingResourceJob { batch, done }
    }

    fn execute_command(&self, command: ResourceCommand) -> Result<(), mem::MemoryError> {
        let mut raw_cmd_buf = self.util_command_pool.begin_single_submit()?;
        let pending_cmd = self
            .process_command(command, Some(&mut raw_cmd_buf))
            .expect("Should be pending");
        let mut pending_job = self.submit_resource_job(vec![pending_cmd], raw_cmd_buf);
        pending_job
            .done
            .blocking_wait()
            .expect("Failed to wait for resource creation");

        assert_eq!(pending_job.batch.len(), 1);
        self.finish_command(pending_job.batch.pop().expect("Expected one cmd"));
        Ok(())
    }

    fn process_commands(&mut self) {
        // Start incoming
        if let Ok(cmd) = self.resource_cmd_receive_queue.try_recv() {
            // Don't create command buffer unless we actually need it
            let mut cmd_buffer = self
                .util_command_pool
                .begin_single_submit()
                .expect("Failed to create command buffer");
            let mut batch = Vec::new();

            if let Some(cmd) = self.process_command(cmd, Some(&mut cmd_buffer)) {
                batch.push(cmd);
            }

            for cmd in self.resource_cmd_receive_queue.try_iter() {
                if let Some(cmd) = self.process_command(cmd, Some(&mut cmd_buffer)) {
                    batch.push(cmd);
                }
            }

            if !batch.is_empty() {
                self.pending_resource_jobs
                    .push(self.submit_resource_job(batch, cmd_buffer));
            }
        }

        // Query finished
        // TODO: Use drain_filter here when not nightly
        let mut i = 0;
        while i < self.pending_resource_jobs.len() {
            if self.pending_resource_jobs[i]
                .done
                .is_signaled()
                .expect("Failed to check fence")
            {
                let PendingResourceJob { batch, done: _done } =
                    self.pending_resource_jobs.remove(i);

                for pending in batch.into_iter() {
                    self.finish_command(pending);
                }
            } else {
                i += 1;
            }
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
        let device = device::Device::new(&instance, &surface)?;

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

        let util_command_pool = command::CommandPool::util(&device)?;
        let descriptor_sets = descriptor::DescriptorSets::new(&device)?;
        let (sender, receiver) = std::sync::mpsc::channel();
        let async_resources = Arc::new(resource::AsyncResources::default());
        let loader_resources = Arc::clone(&async_resources);

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
            resources: Resources::new(descriptor_sets),
            async_resources,
            util_command_pool,
            loader: loader::Loader::new(sender, loader_resources),
            resource_cmd_receive_queue: receiver,
            pending_resource_jobs: Vec::default(),
        })
    }

    pub fn next_frame<'a, 'b: 'a>(&'b mut self) -> Result<Frame<'a>, RenderError> {
        self.process_commands();
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
        let gfx_command_pool = command::CommandPool::graphics(&self.device)?;

        self.image_to_frame_idx[self.swapchain_image_idx as usize] = Some(self.frame_idx);

        Ok(Frame::<'a> {
            renderer: self,
            recorded_command_buffers: Vec::new(),
            gfx_command_pool,
        })
    }

    pub fn submit(&mut self, frame: FinishedFrame) -> Result<(), RenderError> {
        self.process_commands();
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

    pub fn get_descriptor_set(
        &self,
        handle: &Handle<descriptor::DescriptorSet>,
    ) -> Option<&descriptor::DescriptorSet> {
        self.resources
            .descriptor_sets
            .get(handle, self.frame_idx as usize)
    }

    pub fn aspect_ratio(&self) -> f32 {
        let util::Extent2D { width, height } = self.swapchain_extent();

        width as f32 / height as f32
    }

    pub fn swapchain_extent(&self) -> util::Extent2D {
        self.swapchain.info().extent
    }

    pub fn loader(&self) -> loader::Loader {
        self.loader.clone()
    }
}

/// These are functions only used by Frame
impl Renderer {
    fn update_uniform<T: Copy>(
        &mut self,
        h: &BufferHandle<uniform::UniformBuffer>,
        data: &T,
    ) -> Result<(), RenderError> {
        let mut ubuf = self
            .async_resources
            .uniform_buffers
            .get_buffered_mut(h, self.frame_idx as usize)
            .ok_or_else(|| RenderError::InvalidHandle(h.handle().id()))?;

        if let resurs::Async::Available(ubuf) = &mut *ubuf {
            ubuf.update_with(data, h.idx() as u64)
                .map_err(RenderError::UniformBuffer)
        } else {
            log::error!("Tried to update non-existing uniform buffer");
            Ok(())
        }
    }

    fn render_pass(&self) -> &backend::render_pass::RenderPass {
        &self.render_pass
    }

    fn framebuffer(&self) -> &framebuffer::Framebuffer {
        &self.swapchain_framebuffers[self.swapchain_image_idx as usize]
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

use parking_lot::MappedRwLockReadGuard;
macro_rules! impl_buffer_manager {
    ($desc:ty, $resource:ty, $handle:ty, $cmd_enum:ident, $storage:ident) => {
        impl resource::ResourceManager<$desc, $resource, $handle> for Renderer {
            type Error = mem::MemoryError;

            fn get_resource(
                &self,
                handle: &$handle,
            ) -> Option<MappedRwLockReadGuard<'_, resurs::Async<$resource>>> {
                self.async_resources
                    .$storage
                    .get(handle, self.frame_idx as usize)
            }

            fn create_resource_blocking(
                &mut self,
                descriptor: $desc,
            ) -> Result<$handle, Self::Error> {
                let handle = self.async_resources.$storage.allocate(&descriptor);
                let cmd = ResourceCommand::$cmd_enum { descriptor, handle };
                self.execute_command(cmd)?;
                Ok(handle)
            }
        }
    };
}

impl_buffer_manager!(
    mesh::OwningVertexBufferDescriptor,
    mesh::VertexBuffer,
    BufferHandle<mesh::VertexBuffer>,
    CreateVertexBuffer,
    vertex_buffers
);
impl_buffer_manager!(
    mesh::OwningIndexBufferDescriptor,
    mesh::IndexBuffer,
    BufferHandle<mesh::IndexBuffer>,
    CreateIndexBuffer,
    index_buffers
);
impl_buffer_manager!(
    uniform::OwningUniformBufferDescriptor,
    uniform::UniformBuffer,
    BufferHandle<uniform::UniformBuffer>,
    CreateUniformBuffer,
    uniform_buffers
);

macro_rules! impl_resource_manager {
    ($desc:ty, $resource:ty, $handle:ty, $error:ty, $cmd_enum:ident, $storage:ident) => {
        impl resource::ResourceManager<$desc, $resource, $handle> for Renderer {
            type Error = $error;

            fn get_resource(
                &self,
                handle: &$handle,
            ) -> Option<MappedRwLockReadGuard<'_, resurs::Async<$resource>>> {
                self.async_resources.$storage.get(handle)
            }

            // TODO: &self?
            fn create_resource_blocking(
                &mut self,
                descriptor: $desc,
            ) -> Result<$handle, Self::Error> {
                let handle = self.async_resources.$storage.allocate(&descriptor);
                let cmd = ResourceCommand::$cmd_enum { descriptor, handle };
                self.execute_command(cmd)?;
                Ok(handle)
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

macro_rules! impl_pipeline_manager {
    ($desc:ty, $resource:ty, $handle:ty, $error:ty, $cmd_enum:ident, $storage:ident) => {
        impl resource::ResourceManager<$desc, $resource, $handle> for Renderer {
            type Error = $error;

            fn get_resource(
                &self,
                handle: &$handle,
            ) -> Option<MappedRwLockReadGuard<'_, resurs::Async<$resource>>> {
                self.async_resources.$storage.get(handle)
            }

            // TODO: &self?
            fn create_resource_blocking(
                &mut self,
                descriptor: $desc,
            ) -> Result<$handle, Self::Error> {
                let handle = self.async_resources.$storage.allocate(&descriptor);
                let cmd = ResourceCommand::$cmd_enum { descriptor, handle };
                let pending_cmd = self.process_command(cmd, None);
                assert!(
                    pending_cmd.is_none(),
                    "Expected synchronous job for pipeline"
                );
                Ok(handle)
            }
        }
    };
}
impl_pipeline_manager!(
    pipeline::GraphicsPipelineDescriptor,
    pipeline::GraphicsPipeline,
    Handle<pipeline::GraphicsPipeline>,
    pipeline::PipelineError,
    CreatePipeline,
    graphics_pipelines
);
