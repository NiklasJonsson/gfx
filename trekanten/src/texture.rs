use std::path::{Path, PathBuf};

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
use crate::resource::{Cache, Handle, Storage};

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

pub fn load_image<P: AsRef<Path>>(p: &P) -> Result<image::RgbaImage, image::ImageError> {
    let path = p.as_ref();

    log::trace!("Trying to load image from {}", path.display());
    let image = image::open(path)?.to_rgba();

    log::trace!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    Ok(image)
}

#[derive(Debug, Clone)]
pub enum TextureDescriptorTy<'a> {
    File(PathBuf),
    Raw {
        data: &'a [u8],
        extent: util::Extent2D,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MipMaps {
    None,
    Generate,
}

#[derive(Debug, Clone)]
struct DescriptorCommon {
    format: util::Format,
    generate_mipmaps: bool,
}

pub struct TextureDescriptor<'a> {
    ty: TextureDescriptorTy<'a>,
    common: DescriptorCommon,
}

impl<'a> TextureDescriptor<'a> {
    pub fn file(p: PathBuf, format: util::Format, mm: MipMaps) -> Self {
        Self {
            ty: TextureDescriptorTy::File(p),
            common: DescriptorCommon {
                format,
                generate_mipmaps: mm == MipMaps::Generate,
            },
        }
    }

    pub fn raw(data: &'a [u8], extent: util::Extent2D, format: util::Format, mm: MipMaps) -> Self {
        Self {
            ty: TextureDescriptorTy::Raw { data, extent },
            common: DescriptorCommon {
                format,
                generate_mipmaps: mm == MipMaps::Generate,
            },
        }
    }
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
    fn from_raw<'a>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        extent: util::Extent2D,
        common: &DescriptorCommon,
        data: &'a [u8],
    ) -> Result<Self, TextureError> {
        let (image, mip_levels) = if common.generate_mipmaps {
            let mip_levels = (extent.max_dim() as f32).log2().floor() as u32 + 1;
            (
                DeviceImage::device_local_mipmapped(
                    device,
                    queue,
                    command_pool,
                    extent,
                    common.format,
                    mip_levels,
                    data,
                )?,
                mip_levels,
            )
        } else {
            (
                DeviceImage::device_local(
                    device,
                    queue,
                    command_pool,
                    extent,
                    common.format,
                    data,
                )?,
                1,
            )
        };

        let aspect = vk::ImageAspectFlags::COLOR;

        let image_view =
            ImageView::new(device, image.vk_image(), common.format, aspect, mip_levels)?;

        let sampler = Sampler::new(device)?;

        Ok(Self {
            image,
            image_view,
            sampler,
        })
    }

    fn from_file(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        common: &DescriptorCommon,
        path: &Path,
    ) -> Result<Self, TextureError> {
        let image = load_image(&path)?;
        let extent = util::Extent2D {
            width: image.width(),
            height: image.height(),
        };
        let raw_image_data = image.into_raw();
        Self::from_raw(device, queue, command_pool, extent, common, &raw_image_data)
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
    cache: Cache<PathBuf, Texture>,
    storage: Storage<Texture>,
}

impl Textures {
    pub fn new() -> Self {
        Self {
            cache: Cache::default(),
            storage: Storage::<Texture>::new(),
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
        match descriptor.ty {
            TextureDescriptorTy::File(pathbuf) => match self.cache.get(&pathbuf) {
                Some(h) => Ok(h),
                None => {
                    let t = Texture::from_file(
                        device,
                        queue,
                        command_pool,
                        &descriptor.common,
                        &pathbuf,
                    )?;
                    let h = self.storage.add(t);
                    self.cache.add(pathbuf, h);
                    Ok(h)
                }
            },
            TextureDescriptorTy::Raw { data, extent } => {
                let t = Texture::from_raw(
                    device,
                    queue,
                    command_pool,
                    extent,
                    &descriptor.common,
                    data,
                )?;
                Ok(self.storage.add(t))
            }
        }
    }
}
