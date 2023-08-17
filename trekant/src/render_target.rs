use crate::backend::device::Device;
use crate::backend::framebuffer::Framebuffer as FrameBuffer;
use crate::backend::image::ImageView;
use crate::render_pass::RenderPass;
use crate::texture::Texture;

use crate::error::RenderError;
use crate::util;

pub struct RenderTarget {
    pub(crate) inner: FrameBuffer,
}

impl RenderTarget {
    pub fn new(
        device: &Device,
        attachments: &[&Texture],
        render_pass: &RenderPass,
        extent: &util::Extent2D,
    ) -> Result<Self, RenderError> {
        let image_views: Vec<&ImageView> = attachments.iter().map(|t| t.image_view()).collect();
        let inner = FrameBuffer::new(device, &image_views, &render_pass.0, extent)?;
        Ok(Self { inner })
    }
}
