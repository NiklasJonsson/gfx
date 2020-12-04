use ash::vk;

use thiserror::Error;

use super::device_selection::DeviceSuitability;
use crate::surface::SurfaceError;

#[derive(Error, Debug)]
pub enum DeviceCreationError {
    #[error("creation failed: {0}")]
    Creation(vk::Result),
    #[error("Not suitable: {0}")]
    UnsuitableDevice(DeviceSuitability),
    #[error("Missing physical device, is vulkan supported?")]
    MissingPhysicalDevice,
    #[error("Internal vulkan error: {0} {1}")]
    InternalVulkan(vk::Result, &'static str),
    #[error("Surface issue {0}")]
    Surface(#[from] SurfaceError),
}

#[derive(Error, Debug)]
pub enum DeviceError {
    #[error("Device creation failed: {0}")]
    Creation(#[from] DeviceCreationError),
    #[error("vkWaitIdle() failed: {0}")]
    WaitIdle(vk::Result),
    #[error("Allocation failure {0}")]
    Allocation(#[from] vk_mem::error::Error),
}
