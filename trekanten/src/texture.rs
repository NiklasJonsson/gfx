use std::path::PathBuf;

use ash::version::DeviceV1_0;
use ash::vk;

use thiserror::Error;

use crate::command::CommandPool;
use crate::device::Device;
use crate::device::HasVkDevice;
use crate::device::VkDeviceHandle;
use crate::image::{ImageView, ImageViewError};
use crate::mem::DeviceImage;
use crate::mem::MemoryError;
use crate::queue::Queue;
use crate::resource::{CachedStorage, Handle};

use crate::util;

#[derive(Debug, Error)]
pub enum TextureError {
    #[error("Failed to load texture data: {0}")]
    Loading(#[from] image::ImageError),
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),
    #[error("Failed to create sampler: {0}")]
    Sampler(vk::Result),
    #[error("Failed to create image view: {0}")]
    ImageView(#[from] ImageViewError),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TextureDescriptor {
    file_path: PathBuf,
}

impl TextureDescriptor {
    pub fn new(file_path: PathBuf) -> Self {
        Self { file_path }
    }
}

pub fn load_image(desc: &TextureDescriptor) -> Result<image::RgbaImage, image::ImageError> {
    let path = &desc.file_path;

    log::trace!("Trying to load image from {}", path.display());
    let image = image::open(path)?.to_rgba();

    log::trace!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    Ok(image)
}

pub struct Sampler {
    vk_device: VkDeviceHandle,
    vk_sampler: vk::Sampler,
}

impl Sampler {
    pub fn new(device: &Device) -> Result<Self, TextureError> {
        let info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            // From ARM Mali recommendations. 1000 is large enough for any texture
            .max_lod(1000.0);

        let vk_device = device.vk_device();
        let vk_sampler = unsafe {
            vk_device
                .create_sampler(&info, None)
                .map_err(TextureError::Sampler)?
        };

        Ok(Self {
            vk_device,
            vk_sampler,
        })
    }

    pub fn vk_sampler(&self) -> &vk::Sampler {
        &self.vk_sampler
    }
}

impl std::ops::Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.vk_device.destroy_sampler(self.vk_sampler, None);
        }
    }
}

pub struct Texture {
    sampler: Sampler,
    image_view: ImageView,
    image: DeviceImage,
}

impl Texture {
    pub fn create(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &TextureDescriptor,
    ) -> Result<Self, TextureError> {
        let image = load_image(descriptor)?;
        let extents = util::Extent2D {
            width: image.width(),
            height: image.height(),
        };

        let mip_levels = (extents.max_dim() as f32).log2().floor() as u32 + 1;

        let format: util::Format = vk::Format::R8G8B8A8_SRGB.into();

        let raw_image_data = image.into_raw();
        let device_image = DeviceImage::device_local_mipmapped(
            device,
            queue,
            command_pool,
            extents,
            format,
            mip_levels,
            &raw_image_data,
        )?;

        let aspect = vk::ImageAspectFlags::COLOR;

        let image_view =
            ImageView::new(device, device_image.vk_image(), format, aspect, mip_levels)?;

        let sampler = Sampler::new(device)?;

        Ok(Self {
            image: device_image,
            image_view,
            sampler,
        })
    }

    pub fn vk_image(&self) -> &vk::Image {
        &self.image.vk_image()
    }

    pub fn vk_image_view(&self) -> &vk::ImageView {
        &self.image_view.vk_image_view()
    }

    pub fn vk_sampler(&self) -> &vk::Sampler {
        &self.sampler.vk_sampler()
    }
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TEXTURE")
    }
}

#[derive(Default)]
pub struct Textures {
    storage: CachedStorage<TextureDescriptor, Texture>,
}

impl Textures {
    pub fn new() -> Self {
        Self {
            storage: CachedStorage::<TextureDescriptor, Texture>::new(),
        }
    }

    pub fn get(&self, h: &Handle<Texture>) -> Option<&Texture> {
        self.storage.get(h)
    }

    pub fn create(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: TextureDescriptor,
    ) -> Result<Handle<Texture>, TextureError> {
        self.storage.create_or_add(descriptor, |desc| {
            Texture::create(device, queue, command_pool, &desc)
        })
    }
}
