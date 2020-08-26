use crate::*;

use thiserror::Error;

use crate::resource::storage::ID;

#[derive(Debug, Clone, Copy)]
pub enum ResizeReason {
    OutOfDate,
    SubOptimal,
}

#[derive(Debug, Error)]
pub enum RenderError {
    Command(#[from] command::CommandError),
    Instance(#[from] instance::InstanceError),
    DebugUtils(#[from] util::vk_debug::DebugUtilsError),
    Surface(#[from] surface::SurfaceError),
    Device(#[from] device::DeviceError),
    RenderPass(#[from] render_pass::RenderPassError),
    Pipeline(#[from] pipeline::PipelineError),
    Queue(#[from] queue::QueueError),
    Descriptor(#[from] descriptor::DescriptorError),
    ColorBuffer(#[from] color_buffer::ColorBufferError),
    DepthBuffer(#[from] depth_buffer::DepthBufferError),
    Sync(#[from] sync::SyncError),
    Swapchain(swapchain::SwapchainError),
    UniformBuffer(mem::MemoryError),
    VertexBuffer(mem::MemoryError),
    IndexBuffer(mem::MemoryError),
    // TODO: Should this be an error?
    NeedsResize(ResizeReason),
    // TODO: Resource typename here as well
    InvalidHandle(ID),
    MissingUniformBuffersForDescriptor,
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<swapchain::SwapchainError> for RenderError {
    fn from(e: swapchain::SwapchainError) -> Self {
        if let swapchain::SwapchainError::OutOfDate = e {
            RenderError::NeedsResize(ResizeReason::OutOfDate)
        } else {
            RenderError::Swapchain(e)
        }
    }
}
