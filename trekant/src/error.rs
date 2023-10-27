use thiserror::Error;

use crate::backend::*;
use crate::pipeline;
use crate::pipeline_resource;

use crate::backend::MemoryError;

use crate::resource::ID;

#[derive(Debug, Clone, Copy)]
pub enum ResizeReason {
    OutOfDate,
    SubOptimal,
}

#[derive(Debug, Error)]
pub enum RenderError {
    Command(#[from] command::CommandError),
    Instance(#[from] instance::InstanceError),
    DebugUtils(#[from] validation_layers::DebugUtilsError),
    Surface(#[from] surface::SurfaceError),
    Device(#[from] device::DeviceError),
    RenderPass(#[from] render_pass::RenderPassError),
    Pipeline(#[from] pipeline::PipelineError),
    Queue(#[from] queue::QueueError),
    Descriptor(#[from] pipeline_resource::PipelineResourceError),
    Sync(#[from] sync::SyncError),
    Swapchain(swapchain::SwapchainError),
    RenderTarget(#[from] framebuffer::FramebufferError),
    RenderTargetImage(MemoryError),
    RenderTargetImageView(image::ImageViewError),
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
