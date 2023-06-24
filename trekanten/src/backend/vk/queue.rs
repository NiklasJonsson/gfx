use ash::vk;

use thiserror::Error;

use super::device::HasVkDevice;
use super::sync::{Fence, SyncError};

#[derive(Debug, Copy, Clone, Error)]
pub enum QueueError {
    #[error("Failed to submit on queue {0}")]
    Submit(vk::Result),
    #[error("Failed to wait on fence {0}")]
    Fence(#[from] SyncError),
}

#[derive(Clone, Debug)]
pub struct QueueFamily {
    pub index: u32,
    pub props: vk::QueueFamilyProperties,
}
struct VkSubmitFn(
    unsafe extern "system" fn(
        queue: vk::Queue,
        submit_count: u32,
        p_submits: *const vk::SubmitInfo,
        fence: vk::Fence,
    ) -> vk::Result,
);
unsafe impl Send for VkSubmitFn {}

pub struct Queue {
    vk_submit_fn: VkSubmitFn,
    vk_queue: vk::Queue,
    queue_family: QueueFamily,
}

impl Queue {
    pub fn new<D: HasVkDevice>(device: D, queue_family: QueueFamily, vk_queue: vk::Queue) -> Self {
        let vk_device = device.vk_device();
        let vk_submit_fn = VkSubmitFn(vk_device.fp_v1_0().queue_submit);
        Self {
            vk_submit_fn,
            vk_queue,
            queue_family,
        }
    }

    pub fn submit(&self, info: &vk::SubmitInfo, fence: &Fence) -> Result<(), QueueError> {
        let infos = [*info];
        let result = unsafe {
            (self.vk_submit_fn.0)(
                self.vk_queue,
                infos.len() as u32,
                infos.as_ptr(),
                *fence.vk_fence(),
            )
        };

        if result == vk::Result::SUCCESS {
            Ok(())
        } else {
            Err(QueueError::Submit(result))
        }
    }

    pub fn vk_queue(&self) -> &vk::Queue {
        &self.vk_queue
    }

    pub fn family(&self) -> &QueueFamily {
        &self.queue_family
    }
}
