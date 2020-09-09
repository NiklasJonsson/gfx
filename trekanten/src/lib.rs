use ash::vk;

mod color_buffer;
pub mod command;
mod common;
mod depth_buffer;
pub mod descriptor;
mod device;
mod error;
mod framebuffer;
mod image;
mod instance;
mod mem;
pub mod mesh;
pub mod pipeline;
mod queue;
mod render_pass;
pub mod resource;
mod surface;
mod swapchain;
mod sync;
pub mod texture;
pub mod uniform;
pub mod util;
pub mod vertex;

pub use error::RenderError;
pub use error::ResizeReason;
pub use mem::BufferHandle;
pub use resource::Handle;
pub use resource::ResourceManager;

use ash::version::DeviceV1_0;
use common::MAX_FRAMES_IN_FLIGHT;
use device::HasVkDevice;

// Notes:
// We can have N number of swapchain images, it depends on the backing presentation implementation.
// Generally, we are aiming for three images + MAILBOX (render one and use the latest of the two waiting)
//
// We use MAX_FRAMES_IN_FLIGHT (2, hardcoded atm) frames in flight at once. This allows us to start the next frame directly after we render.
// Whenever next_frame() is called, it can be thought of as binding one of the two frames to a particular swapchain image.
// All rendering in that frame will be done on that swapchain image/framebuffer.

pub struct FrameSynchronization {
    pub image_available: sync::Semaphore,
    pub render_done: sync::Semaphore,
    pub in_flight: sync::Fence,
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
        })
    }
}

pub struct Frame {
    frame_idx: u32,
    swapchain_image_idx: u32,
    recorded_command_buffers: Vec<vk::CommandBuffer>,
    gfx_command_pool: command::CommandPool,
}

impl Frame {
    pub fn new_command_buffer(&self) -> Result<command::CommandBuffer, command::CommandError> {
        self.gfx_command_pool
            .create_command_buffer(command::CommandBufferSubmission::Single)
    }

    pub fn add_command_buffer(&mut self, cmd_buffer: command::CommandBuffer) {
        self.recorded_command_buffers
            .push(*cmd_buffer.vk_command_buffer());
    }
}

pub struct Renderer {
    // Resources
    graphics_pipelines: pipeline::GraphicsPipelines,
    vertex_buffers: resource::Storage<mesh::VertexBuffer>,
    index_buffers: resource::Storage<mesh::IndexBuffer>,
    uniform_buffers: uniform::UniformBuffers,
    descriptor_sets: descriptor::DescriptorSets,
    textures: texture::Textures,

    // Swapchain-related
    // TODO: Could render pass be a abstracted as forward-renderer?
    render_pass: render_pass::RenderPass,
    swapchain_framebuffers: Vec<framebuffer::Framebuffer>,
    depth_buffer: depth_buffer::DepthBuffer,
    color_buffer: color_buffer::ColorBuffer,
    swapchain: swapchain::Swapchain,
    swapchain_image_idx: u32, // TODO: Bake this into the swapchain?
    image_to_frame_idx: Vec<Option<u32>>,

    util_command_pool: command::CommandPool,

    // Needs to be kept-alive
    _debug_utils: util::vk_debug::DebugUtils,

    frame_synchronization: [FrameSynchronization; MAX_FRAMES_IN_FLIGHT],
    frame_idx: u32,
    frames: [Option<Frame>; MAX_FRAMES_IN_FLIGHT],

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
    render_pass: render_pass::RenderPass,
}

fn create_swapchain_and_co(
    instance: &instance::Instance,
    device: &device::Device,
    surface: &surface::Surface,
    extent: &util::Extent2D,
    old: Option<&swapchain::Swapchain>,
) -> Result<SwapchainAndCo, RenderError> {
    let msaa_sample_count = device.max_msaa_sample_count();
    let swapchain = swapchain::Swapchain::new(&instance, &device, &surface, &extent, old)?;
    let render_pass =
        render_pass::RenderPass::new(&device, swapchain.info().format, msaa_sample_count)?;

    let image_to_frame_idx: Vec<Option<u32>> = (0..swapchain.num_images()).map(|_| None).collect();
    let depth_buffer = depth_buffer::DepthBuffer::new(device, extent, msaa_sample_count)?;
    let color_buffer = color_buffer::ColorBuffer::new(
        device,
        swapchain.info().format.into(),
        extent,
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

        let frames = [None, None];
        let frame_synchronization = [
            FrameSynchronization::new(&device)?,
            FrameSynchronization::new(&device)?,
        ];

        let util_command_pool = command::CommandPool::util(&device)?;
        let descriptor_sets = descriptor::DescriptorSets::new(&device)?;

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
            frames,
            swapchain_image_idx: 0,
            _debug_utils,
            graphics_pipelines: Default::default(),
            vertex_buffers: Default::default(),
            index_buffers: Default::default(),
            uniform_buffers: Default::default(),
            textures: Default::default(),
            descriptor_sets,
            util_command_pool,
        })
    }

    pub fn next_frame(&mut self) -> Result<Frame, RenderError> {
        let frame_sync = &self.frame_synchronization[self.frame_idx as usize];
        frame_sync.in_flight.blocking_wait()?;

        self.swapchain_image_idx = self
            .swapchain
            .acquire_next_image(Some(&frame_sync.image_available))?;

        // This means that we received an image that might be in the process of rendering
        if let Some(frame_idx) = self.image_to_frame_idx[self.swapchain_image_idx as usize] {
            self.frame_synchronization[frame_idx as usize]
                .in_flight
                .blocking_wait()?;
        }

        // This will drop the frame that resided here previously
        let _ = std::mem::replace(&mut self.frames[self.frame_idx as usize], None);

        let gfx_command_pool = command::CommandPool::graphics(&self.device)?;

        self.image_to_frame_idx[self.swapchain_image_idx as usize] = Some(self.frame_idx);

        Ok(Frame {
            frame_idx: self.frame_idx,
            swapchain_image_idx: self.swapchain_image_idx,
            recorded_command_buffers: Vec::new(),
            gfx_command_pool,
        })
    }

    pub fn submit(&mut self, frame: Frame) -> Result<(), RenderError> {
        assert_eq!(frame.frame_idx, self.frame_idx, "Mismatching frame indexes");

        // Make sure that this is captured before any early returns. If this function returns
        // without having extended the lifetime of frame, it might be dropped while it's command
        // buffers are still in use.
        self.frames[self.frame_idx as usize] = Some(frame);
        let frame = self.frames[self.frame_idx as usize].as_ref().unwrap();

        let frame_sync = &self.frame_synchronization[self.frame_idx as usize];
        let vk_wait_sems = [*frame_sync.image_available.vk_semaphore()];
        let wait_dst_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let vk_sig_sems = [*frame_sync.render_done.vk_semaphore()];

        let info = vk::SubmitInfo::builder()
            .wait_semaphores(&vk_wait_sems)
            .wait_dst_stage_mask(&wait_dst_mask)
            .signal_semaphores(&vk_sig_sems)
            .command_buffers(&frame.recorded_command_buffers);

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

    pub fn render_pass(&self) -> &render_pass::RenderPass {
        &self.render_pass
    }

    pub fn swapchain_extent(&self) -> util::Extent2D {
        self.swapchain.info().extent
    }

    pub fn framebuffer(&self, frame: &Frame) -> &framebuffer::Framebuffer {
        &self.swapchain_framebuffers[frame.swapchain_image_idx as usize]
    }

    fn recreate_pipelines(&mut self) -> Result<(), RenderError> {
        log::trace!("Recreating pipelines with {}", self.swapchain_extent());
        self.graphics_pipelines.recreate_all(
            &self.device,
            self.swapchain_extent(),
            &self.render_pass,
        )?;
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

        self.recreate_pipelines()?;

        Ok(())
    }

    pub fn update_uniform<T>(
        &mut self,
        h: &BufferHandle<uniform::UniformBuffer>,
        data: &T,
    ) -> Result<(), RenderError> {
        let ubuf = self
            .uniform_buffers
            .get_mut(h.handle(), self.frame_idx as usize)
            .ok_or_else(|| RenderError::InvalidHandle(h.handle().id()))?;

        ubuf.update_with(data, h.offset())
            .map_err(RenderError::UniformBuffer)
    }

    pub fn get_descriptor_set(
        &self,
        handle: &Handle<descriptor::DescriptorSet>,
    ) -> Option<&descriptor::DescriptorSet> {
        self.descriptor_sets.get(handle, self.frame_idx as usize)
    }

    pub fn aspect_ratio(&self) -> f32 {
        let util::Extent2D { width, height } = self.swapchain_extent();

        width as f32 / height as f32
    }
}

// TODO: Everything in this impl needs to be refactored
impl Renderer {
    fn get_uniform_buffers(
        &self,
        handle: &Handle<uniform::UniformBuffer>,
    ) -> Option<&[uniform::UniformBuffer; MAX_FRAMES_IN_FLIGHT]> {
        self.uniform_buffers.get_all(handle)
    }

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
        self.descriptor_sets
            .alloc(bindings)
            .expect("Failed to alloc")
    }
}

impl
    resource::ResourceManager<
        pipeline::GraphicsPipelineDescriptor,
        pipeline::GraphicsPipeline,
        pipeline::PipelineError,
    > for Renderer
{
    type Handle = resource::Handle<pipeline::GraphicsPipeline>;

    fn get_resource(&self, handle: &Self::Handle) -> Option<&pipeline::GraphicsPipeline> {
        self.graphics_pipelines.get(handle)
    }

    fn create_resource(
        &mut self,
        descriptor: pipeline::GraphicsPipelineDescriptor,
    ) -> Result<Self::Handle, pipeline::PipelineError> {
        self.graphics_pipelines.create(
            &self.device,
            descriptor,
            self.swapchain_extent(),
            &self.render_pass,
        )
    }
}

impl<'a>
    resource::ResourceManager<
        mesh::VertexBufferDescriptor<'a>,
        mesh::VertexBuffer,
        mem::MemoryError,
    > for Renderer
{
    type Handle = resource::Handle<mesh::VertexBuffer>;

    fn get_resource(&self, handle: &Self::Handle) -> Option<&mesh::VertexBuffer> {
        self.vertex_buffers.get(handle)
    }

    fn create_resource(
        &mut self,
        descriptor: mesh::VertexBufferDescriptor<'a>,
    ) -> Result<Self::Handle, mem::MemoryError> {
        let queue = self.device.util_queue();
        let new =
            mesh::VertexBuffer::create(&self.device, queue, &self.util_command_pool, &descriptor)?;

        Ok(self.vertex_buffers.add(new))
    }
}

impl<'a>
    resource::ResourceManager<mesh::IndexBufferDescriptor<'a>, mesh::IndexBuffer, mem::MemoryError>
    for Renderer
{
    type Handle = resource::Handle<mesh::IndexBuffer>;

    fn get_resource(&self, handle: &Self::Handle) -> Option<&mesh::IndexBuffer> {
        self.index_buffers.get(handle)
    }

    fn create_resource(
        &mut self,
        descriptor: mesh::IndexBufferDescriptor<'a>,
    ) -> Result<Self::Handle, mem::MemoryError> {
        let queue = self.device.util_queue();
        let new =
            mesh::IndexBuffer::create(&self.device, queue, &self.util_command_pool, &descriptor)?;

        Ok(self.index_buffers.add(new))
    }
}

impl<'a, T>
    resource::ResourceManager<
        uniform::UniformBufferDescriptor<'a, T>,
        uniform::UniformBuffer,
        mem::MemoryError,
    > for Renderer
{
    type Handle = mem::BufferHandle<uniform::UniformBuffer>;
    fn get_resource(&self, handle: &Self::Handle) -> Option<&uniform::UniformBuffer> {
        self.uniform_buffers
            .get(handle.handle(), self.frame_idx as usize)
    }

    fn create_resource(
        &mut self,
        descriptor: uniform::UniformBufferDescriptor<'a, T>,
    ) -> Result<Self::Handle, mem::MemoryError> {
        let queue = self.device.util_queue();
        let resource_handle = self.uniform_buffers.create(
            &self.device,
            queue,
            &self.util_command_pool,
            &descriptor,
        )?;

        // Frame index doesn't matter, stats are the same
        let buf = self
            .uniform_buffers
            .get(&resource_handle, 0)
            .expect("Just added this...");
        unsafe {
            Ok(Self::Handle::from_strided_buffer(
                resource_handle,
                0,
                buf.n_elems(),
                buf.elem_size(),
                buf.stride(),
            ))
        }
    }
}

impl<'a>
    resource::ResourceManager<texture::TextureDescriptor, texture::Texture, texture::TextureError>
    for Renderer
{
    type Handle = resource::Handle<texture::Texture>;
    fn get_resource(&self, handle: &Self::Handle) -> Option<&texture::Texture> {
        self.textures.get(handle)
    }

    fn create_resource(
        &mut self,
        descriptor: texture::TextureDescriptor,
    ) -> Result<Self::Handle, texture::TextureError> {
        let queue = self.device.util_queue();
        self.textures
            .create(&self.device, queue, &self.util_command_pool, descriptor)
    }
}
