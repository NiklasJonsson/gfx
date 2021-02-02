use std::path::{Path, PathBuf};

use ash::version::DeviceV1_0;
use ash::vk;

use thiserror::Error;

use crate::backend::{AllocatorHandle, HasVkDevice, VkDeviceHandle};
use crate::command::CommandBuffer;
use crate::image::{ImageView, ImageViewError};
use crate::mem::DeviceBuffer;
use crate::mem::DeviceImage;
use crate::mem::MemoryError;
use crate::resource::{Cache, Handle, Storage};
use crate::util;

use std::sync::Arc;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MipMaps {
    None,
    Generate,
}

#[derive(Debug, Clone)]
struct DescriptorCommon {
    format: util::Format,
    mipmaps: MipMaps,
}

#[derive(Clone, Debug)]
pub enum TextureDescriptorTy {
    File(PathBuf),
    Raw {
        data: Arc<util::ByteBuffer>,
        extent: util::Extent2D,
    },
}

#[derive(Clone, Debug)]
pub struct TextureDescriptor {
    ty: TextureDescriptorTy,
    common: DescriptorCommon,
}

impl TextureDescriptor {
    pub fn file(p: PathBuf, format: util::Format, mipmaps: MipMaps) -> Self {
        Self {
            ty: TextureDescriptorTy::File(p),
            common: DescriptorCommon { format, mipmaps },
        }
    }

    pub fn from_vec(
        data: Vec<u8>,
        extent: util::Extent2D,
        format: util::Format,
        mipmaps: MipMaps,
    ) -> Self {
        Self {
            ty: TextureDescriptorTy::Raw {
                data: Arc::new(unsafe { util::ByteBuffer::from_vec(data) }),
                extent,
            },
            common: DescriptorCommon { format, mipmaps },
        }
    }

    pub fn enqueue<D: HasVkDevice>(
        &self,
        allocator: &AllocatorHandle,
        device: &D,
        command_buffer: &mut CommandBuffer,
    ) -> Result<(Texture, DeviceBuffer), TextureError> {
        match &self.ty {
            TextureDescriptorTy::File(pathbuf) => {
                Texture::from_file(device, allocator, command_buffer, &self.common, &pathbuf)
            }
            TextureDescriptorTy::Raw { data, extent } => Texture::from_raw(
                device,
                allocator,
                command_buffer,
                *extent,
                &self.common,
                &data,
            ),
        }
    }
}

pub struct Sampler {
    vk_device: VkDeviceHandle,
    vk_sampler: vk::Sampler,
}

// TODO: Support other border color/address mode
impl Sampler {
    pub fn new<D: HasVkDevice>(device: &D) -> Result<Self, TextureError> {
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
    fn from_raw<'a, D: HasVkDevice>(
        device: &D,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        extent: util::Extent2D,
        common: &DescriptorCommon,
        data: &'a [u8],
    ) -> Result<(Self, DeviceBuffer), TextureError> {
        let ((image, staging), mip_levels) = if let MipMaps::Generate = common.mipmaps {
            let mip_levels = (extent.max_dim() as f32).log2().floor() as u32 + 1;
            (
                DeviceImage::device_local_mipmapped(
                    &allocator,
                    command_buffer,
                    extent,
                    common.format,
                    mip_levels,
                    data,
                )?,
                mip_levels,
            )
        } else {
            (
                DeviceImage::device_local(&allocator, command_buffer, extent, common.format, data)?,
                1,
            )
        };

        let aspect = vk::ImageAspectFlags::COLOR;

        let image_view =
            ImageView::new(device, image.vk_image(), common.format, aspect, mip_levels)?;

        let sampler = Sampler::new(device)?;

        Ok((
            Self {
                image,
                image_view,
                sampler,
            },
            staging,
        ))
    }

    fn from_file<D: HasVkDevice>(
        device: &D,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        common: &DescriptorCommon,
        path: &Path,
    ) -> Result<(Self, DeviceBuffer), TextureError> {
        let image = load_image(&path)?;
        let extent = util::Extent2D {
            width: image.width(),
            height: image.height(),
        };
        let raw_image_data = image.into_raw();
        Self::from_raw(
            device,
            allocator,
            command_buffer,
            extent,
            common,
            &raw_image_data,
        )
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

pub struct TextureStorage<T> {
    cache: Cache<PathBuf, T>,
    storage: Storage<T>,
}

impl<T> TextureStorage<T> {
    pub fn new() -> Self {
        Self {
            cache: Cache::default(),
            storage: Storage::<T>::new(),
        }
    }

    pub fn get(&self, handle: &Handle<T>) -> Option<&T> {
        self.storage.get(handle)
    }

    pub fn get_mut(&mut self, handle: &Handle<T>) -> Option<&mut T> {
        self.storage.get_mut(handle)
    }

    pub fn add(&mut self, descriptor: &TextureDescriptor, t: T) -> Handle<T> {
        let h = self.storage.add(t);

        if let TextureDescriptorTy::File(path) = &descriptor.ty {
            self.cache.add(path.clone(), h);
        }

        h
    }

    pub fn cached(&self, descriptor: &TextureDescriptor) -> Option<Handle<T>> {
        if let TextureDescriptorTy::File(path) = &descriptor.ty {
            self.cache.get(&path).cloned()
        } else {
            None
        }
    }
}
impl<T> TextureStorage<Async<T>> {
    pub fn allocate(&mut self, _desc: &TextureDescriptor) -> Handle<Async<T>> {
        self.storage.add(Async::Pending)
    }
}

impl<T> Default for TextureStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub type Textures = TextureStorage<Texture>;
use crate::resource::Async;
pub type AsyncTextures = TextureStorage<Async<Texture>>;
