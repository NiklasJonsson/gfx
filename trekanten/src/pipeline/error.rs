use ash::vk;

use thiserror::Error;

use crate::spirv::SpirvError;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Could not create {1}: {0}")]
    VulkanObjectCreation(vk::Result, &'static str),
    #[error("Missing required arg: {0}")]
    MissingArg(&'static str),
    #[error("Spirv reflection failed: {0}")]
    Reflection(#[from] SpirvError),
}
