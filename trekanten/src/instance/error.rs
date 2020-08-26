use ash::vk;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum InstanceError {
    #[error("Instance creation failed: {0}")]
    Creation(vk::Result),
    #[error("Missing a vulkan extension: {0}")]
    MissingExtension(String),
    #[error("Failed to load instance: {0:?}")]
    LoadError(Vec<&'static str>),
    #[error("Internal vulkan error: {0} {1}")]
    InternalVulkan(vk::Result, &'static str),
}

impl From<ash::InstanceError> for InstanceError {
    fn from(e: ash::InstanceError) -> Self {
        match e {
            ash::InstanceError::VkError(r) => InstanceError::Creation(r),
            ash::InstanceError::LoadError(v) => InstanceError::LoadError(v),
        }
    }
}
