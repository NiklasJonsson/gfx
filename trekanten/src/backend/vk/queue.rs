use ash::version::DeviceV1_0;
use ash::vk;

use thiserror::Error;

use super::command::CommandBuffer;
use super::device::{HasVkDevice, VkDeviceHandle};
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

#[derive(Clone, Debug)]
pub struct QueueFamilies {
    pub graphics: QueueFamily,
    pub present: QueueFamily,
}

#[derive(Clone)]
pub struct Queue {
    vk_device: VkDeviceHandle,
    vk_queue: vk::Queue,
}

impl Queue {
    pub fn new<D: HasVkDevice>(device: D, vk_queue: vk::Queue) -> Self {
        Self {
            vk_device: device.vk_device(),
            vk_queue,
        }
    }

    pub fn submit(&self, info: &vk::SubmitInfo, fence: &Fence) -> Result<(), QueueError> {
        let infos = [*info];
        unsafe {
            self.vk_device
                .queue_submit(self.vk_queue, &infos, *fence.vk_fence())
                .map_err(QueueError::Submit)?;
        }

        Ok(())
    }

    pub fn submit_and_wait(&self, cmd_buf: &CommandBuffer) -> Result<(), QueueError> {
        let bufs = [*cmd_buf.vk_command_buffer()];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&bufs);

        let copied = Fence::unsignaled(&self.vk_device)?;
        self.submit(&submit_info, &copied)?;

        // TODO: Async
        copied.blocking_wait()?;

        Ok(())
    }

    pub fn vk_queue(&self) -> &vk::Queue {
        &self.vk_queue
    }
}
