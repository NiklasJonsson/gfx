use ash::vk;

use thiserror::Error;

use crate::backend;

use crate::util;
use backend::device::Device;
use backend::image::{Image, ImageView, ImageViewError};
use backend::MemoryError;

#[derive(Debug, Error)]
pub enum ColorBufferError {
    #[error("Color buffer memory error: {0}")]
    Memory(#[from] MemoryError),
    #[error("Color buffer image view error: {0}")]
    ImageView(#[from] ImageViewError),
}

pub struct ColorBuffer {
    _image: Image,
    image_view: ImageView,
    _format: util::Format,
}

impl ColorBuffer {
    pub fn new(
        device: &Device,
        format: util::Format,
        extents: &util::Extent2D,
        msaa_sample_count: vk::SampleCountFlags,
    ) -> Result<Self, ColorBufferError> {
        let usage =
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let props = vk_mem::MemoryUsage::GpuOnly;
        let mip_levels = 1; // No mip maps
        let _image = Image::empty_2d(
            &device.allocator(),
            *extents,
            format,
            usage,
            props,
            mip_levels,
            msaa_sample_count,
        )?;
        let image_view = ImageView::new(
            device,
            _image.vk_image(),
            format,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;
        Ok(Self {
            _image,
            image_view,
            _format: format,
        })
    }

    pub fn image_view(&self) -> &ImageView {
        &self.image_view
    }
}
