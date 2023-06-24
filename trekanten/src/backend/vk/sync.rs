use ash::vk;

use thiserror::Error;

use super::device::{HasVkDevice, VkDeviceHandle};

#[derive(Debug, Copy, Clone, Error)]
pub enum SyncError {
    #[error("Semaphore creation failed {0}")]
    SemaphoreCreation(vk::Result),
    #[error("Febce creation failed {0}")]
    FenceCreation(vk::Result),
    #[error("Couldn't wait on fence {0}")]
    FenceAwait(vk::Result),
    #[error("Couldn't reset fence {0}")]
    FenceReset(vk::Result),
    #[error("Couldn't query fence {0}")]
    FenceQuery(vk::Result),
}

#[derive(Clone)]
pub struct Semaphore {
    vk_semaphore: vk::Semaphore,
    vk_device: VkDeviceHandle,
}

impl std::ops::Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.vk_device.destroy_semaphore(self.vk_semaphore, None);
        }
    }
}

impl Semaphore {
    pub fn new<D: HasVkDevice>(device: &D) -> Result<Self, SyncError> {
        let vk_device = device.vk_device();
        let info = vk::SemaphoreCreateInfo::default();

        let vk_semaphore = unsafe {
            vk_device
                .create_semaphore(&info, None)
                .map_err(SyncError::SemaphoreCreation)?
        };

        Ok(Self {
            vk_device,
            vk_semaphore,
        })
    }

    pub fn vk_semaphore(&self) -> &vk::Semaphore {
        &self.vk_semaphore
    }
}

#[derive(Clone)]
pub struct Fence {
    vk_fence: vk::Fence,
    vk_device: VkDeviceHandle,
}

impl std::ops::Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.vk_device.destroy_fence(self.vk_fence, None);
        }
    }
}

impl Fence {
    fn new<D: HasVkDevice>(device: &D, flags: vk::FenceCreateFlags) -> Result<Self, SyncError> {
        let vk_device = device.vk_device();
        let info = vk::FenceCreateInfo {
            flags,
            ..Default::default()
        };

        let vk_fence = unsafe {
            vk_device
                .create_fence(&info, None)
                .map_err(SyncError::FenceCreation)?
        };

        Ok(Self {
            vk_device,
            vk_fence,
        })
    }

    pub fn signaled<D: HasVkDevice>(device: &D) -> Result<Self, SyncError> {
        Self::new(device, vk::FenceCreateFlags::SIGNALED)
    }

    pub fn unsignaled<D: HasVkDevice>(device: &D) -> Result<Self, SyncError> {
        Self::new(device, vk::FenceCreateFlags::empty())
    }

    pub fn vk_fence(&self) -> &vk::Fence {
        &self.vk_fence
    }

    pub fn blocking_wait(&self) -> Result<(), SyncError> {
        let fences = [self.vk_fence];
        unsafe {
            self.vk_device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(SyncError::FenceAwait)?;
        }

        Ok(())
    }

    pub fn is_signaled(&self) -> Result<bool, SyncError> {
        unsafe {
            self.vk_device
                .get_fence_status(self.vk_fence)
                .map_err(SyncError::FenceQuery)
        }
    }

    pub fn reset(&self) -> Result<(), SyncError> {
        let fences = [self.vk_fence];
        unsafe {
            self.vk_device
                .reset_fences(&fences)
                .map_err(SyncError::FenceReset)?;
        }

        Ok(())
    }
}
