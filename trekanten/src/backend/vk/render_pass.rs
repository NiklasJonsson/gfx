use ash::version::DeviceV1_0;
use ash::vk;

use thiserror::Error;

use super::device::{Device, HasVkDevice, VkDeviceHandle};

#[derive(Clone, Error, Debug)]
pub enum RenderPassError {
    #[error("Render pass creation failed")]
    Creation(vk::Result),
}

pub struct RenderPass {
    vk_device: VkDeviceHandle,
    vk_render_pass: vk::RenderPass,
    msaa_sample_count: vk::SampleCountFlags,
}

impl std::ops::Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.vk_device
                .destroy_render_pass(self.vk_render_pass, None);
        }
    }
}

impl RenderPass {
    pub fn new(
        device: &Device,
        create_info: &vk::RenderPassCreateInfo,
    ) -> Result<Self, RenderPassError> {
        let vk_device = device.vk_device();

        let vk_render_pass = unsafe {
            vk_device
                .create_render_pass(create_info, None)
                .map_err(RenderPassError::Creation)?
        };

        let attachments = unsafe {
            std::slice::from_raw_parts(
                create_info.p_attachments,
                create_info.attachment_count as usize,
            )
        };

        let msaa_sample_count = attachments
            .iter()
            .fold(vk::SampleCountFlags::TYPE_1, |acc, att| {
                std::cmp::max(acc, att.samples)
            });

        Ok(Self {
            vk_device,
            vk_render_pass,
            msaa_sample_count,
        })
    }

    pub fn vk_render_pass(&self) -> &vk::RenderPass {
        &self.vk_render_pass
    }

    pub fn msaa_sample_count(&self) -> vk::SampleCountFlags {
        self.msaa_sample_count
    }
}
