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
use crate::resource::{Handle, Storage};
use crate::util;
use util::Extent2D;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipMaps {
    None,
    Generate,
}

#[derive(Debug, Clone)]
pub struct DescriptorCommon {}

bitflags::bitflags! {
    pub struct TextureUsage: u8 {
        const DEPTH_STENCIL_ATTACHMENT = 0b1;
        const COLOR_ATTACHMENT = 0b10;
        const TRANSFER_SRC = 0b100;
        const TRANSFER_DST = 0b1000;
    }
}

macro_rules! impl_flag_mapping {
    ($result:ident, $input:ident, $flag:ident) => {
        if !($input & TextureUsage::$flag).is_empty() {
            $result |= vk::ImageUsageFlags::$flag;
        }
    };
    ($result:ident, $input:ident, $flag:ident, $($rest:ident),*) => {
        impl_flag_mapping!($result, $input, $flag);
        impl_flag_mapping!($result, $input, $($rest),*)
    };
}

impl From<TextureUsage> for vk::ImageUsageFlags {
    fn from(o: TextureUsage) -> vk::ImageUsageFlags {
        let mut ret = vk::ImageUsageFlags::empty();
        impl_flag_mapping!(
            ret,
            o,
            DEPTH_STENCIL_ATTACHMENT,
            COLOR_ATTACHMENT,
            TRANSFER_SRC,
            TRANSFER_DST
        );

        ret
    }
}

#[derive(Clone, Debug)]
pub enum TextureDescriptor {
    File {
        path: PathBuf,
        format: util::Format,
        mipmaps: MipMaps,
    },
    Raw {
        data: Arc<util::ByteBuffer>,
        extent: Extent2D,
        format: util::Format,
        mipmaps: MipMaps,
    },
    Empty {
        extent: Extent2D,
        usage: TextureUsage,
        format: util::Format,
        sampler: SamplerDescriptor,
    },
}

impl TextureDescriptor {
    pub fn mipmaps(&self) -> MipMaps {
        match self {
            Self::File { mipmaps, .. } | Self::Raw { mipmaps, .. } => *mipmaps,
            Self::Empty { .. } => MipMaps::None,
        }
    }

    pub fn file(p: PathBuf, format: util::Format, mipmaps: MipMaps) -> Self {
        Self::File {
            path: p,
            format,
            mipmaps,
        }
    }

    pub fn from_vec(
        data: Vec<u8>,
        extent: Extent2D,
        format: util::Format,
        mipmaps: MipMaps,
    ) -> Self {
        Self::Raw {
            data: Arc::new(unsafe { util::ByteBuffer::from_vec(data) }),
            extent,
            format,
            mipmaps,
        }
    }

    pub(crate) fn needs_command_buffer(&self) -> bool {
        if let TextureDescriptor::Empty { .. } = self {
            false
        } else {
            true
        }
    }

    pub fn enqueue<D: HasVkDevice>(
        &self,
        allocator: &AllocatorHandle,
        device: &D,
        command_buffer: &mut CommandBuffer,
    ) -> Result<(Texture, DeviceBuffer), TextureError> {
        match self {
            TextureDescriptor::File {
                path,
                format,
                mipmaps,
            } => {
                let image = load_image(&path)?;
                let extent = Extent2D {
                    width: image.width(),
                    height: image.height(),
                };
                let raw_image_data = image.into_raw();
                Texture::from_raw(
                    device,
                    allocator,
                    command_buffer,
                    extent,
                    *format,
                    *mipmaps,
                    &raw_image_data,
                )
            }
            TextureDescriptor::Raw {
                data,
                extent,
                format,
                mipmaps,
            } => Texture::from_raw(
                device,
                allocator,
                command_buffer,
                *extent,
                *format,
                *mipmaps,
                &data,
            ),
            _ => unreachable!("This should not be created with a command buffer"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Filter {
    Linear,
    Nearest,
}

impl From<Filter> for vk::Filter {
    fn from(o: Filter) -> vk::Filter {
        match o {
            Filter::Linear => vk::Filter::LINEAR,
            Filter::Nearest => vk::Filter::NEAREST,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

impl From<SamplerAddressMode> for vk::SamplerAddressMode {
    fn from(o: SamplerAddressMode) -> vk::SamplerAddressMode {
        use vk::SamplerAddressMode as VSAM;
        use SamplerAddressMode as SAM;
        match o {
            SAM::Repeat => VSAM::REPEAT,
            SAM::MirroredRepeat => VSAM::MIRRORED_REPEAT,
            SAM::ClampToEdge => VSAM::CLAMP_TO_EDGE,
            SAM::ClampToBorder => VSAM::CLAMP_TO_BORDER,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BorderColor {
    FloatTransparentBlack,
    IntTransparentBlack,
    FloatOpaqueBlack,
    IntOpaqueBlack,
    FloatOpaqueWhite,
    IntOpaqueWhite,
}

impl From<BorderColor> for vk::BorderColor {
    fn from(o: BorderColor) -> vk::BorderColor {
        use vk::BorderColor as VBC;
        use BorderColor as BC;
        match o {
            BC::FloatOpaqueBlack => VBC::FLOAT_OPAQUE_BLACK,
            BC::FloatOpaqueWhite => VBC::FLOAT_OPAQUE_WHITE,
            BC::FloatTransparentBlack => VBC::FLOAT_TRANSPARENT_BLACK,
            BC::IntOpaqueBlack => VBC::INT_OPAQUE_BLACK,
            BC::IntOpaqueWhite => VBC::INT_OPAQUE_WHITE,
            BC::IntTransparentBlack => VBC::INT_TRANSPARENT_BLACK,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerDescriptor {
    pub filter: Filter,
    pub address_mode: SamplerAddressMode,
    pub max_anisotropy: Option<f32>,
    pub border_color: BorderColor,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            filter: Filter::Linear,
            address_mode: SamplerAddressMode::Repeat,
            max_anisotropy: Some(16.0),
            border_color: BorderColor::IntOpaqueBlack,
        }
    }
}

pub struct Sampler {
    vk_device: VkDeviceHandle,
    vk_sampler: vk::Sampler,
}

impl Sampler {
    pub fn new<D: HasVkDevice>(device: &D, desc: &SamplerDescriptor) -> Result<Self, TextureError> {
        let filter = vk::Filter::from(desc.filter);
        let address_mode = vk::SamplerAddressMode::from(desc.address_mode);
        let border_color = vk::BorderColor::from(desc.border_color);
        let mut info = vk::SamplerCreateInfo::builder()
            .mag_filter(filter)
            .min_filter(filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
            .border_color(border_color)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            // From ARM Mali recommendations. 1000 is large enough for any texture
            .max_lod(1000.0)
            .build();

        if let Some(anisotropy) = desc.max_anisotropy {
            info.max_anisotropy = anisotropy;
            info.anisotropy_enable = vk::TRUE;
        }

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

pub fn mip_levels_for(e: Extent2D) -> u32 {
    (e.max_dim() as f32).log2().floor() as u32 + 1
}

pub struct Texture {
    sampler: Sampler,
    image_view: ImageView,
    image: DeviceImage,
}

impl Texture {
    pub(crate) fn create_no_cmds<D: HasVkDevice>(
        device: &D,
        allocator: &AllocatorHandle,
        descriptor: &TextureDescriptor,
    ) -> Result<Self, TextureError> {
        if let TextureDescriptor::Empty {
            extent,
            format,
            usage,
            sampler: sampler_descriptor,
        } = descriptor
        {
            let image_usage = vk::ImageUsageFlags::from(*usage) | vk::ImageUsageFlags::SAMPLED;
            let mem_usage = vk_mem::MemoryUsage::GpuOnly;
            let mip_levels = 1;
            let sample_count = vk::SampleCountFlags::TYPE_1;
            let aspect_mask = if usage.contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let image = DeviceImage::empty_2d(
                allocator,
                *extent,
                *format,
                image_usage,
                mem_usage,
                mip_levels,
                sample_count,
            )?;
            let image_view =
                ImageView::new(device, image.vk_image(), *format, aspect_mask, mip_levels)?;
            let sampler = Sampler::new(device, &sampler_descriptor)?;
            Ok(Self {
                image,
                image_view,
                sampler,
            })
        } else {
            unreachable!("This needs a command buffer");
        }
    }

    pub(crate) fn from_device_image<D: HasVkDevice>(
        device: &D,
        image: DeviceImage,
        format: util::Format,
        mip_levels: u32,
    ) -> Result<Self, TextureError> {
        let aspect = vk::ImageAspectFlags::COLOR;

        let image_view = ImageView::new(device, image.vk_image(), format, aspect, mip_levels)?;

        let sampler = Sampler::new(device, &SamplerDescriptor::default())?;

        Ok(Self {
            image,
            image_view,
            sampler,
        })
    }
    fn from_raw<'a, D: HasVkDevice>(
        device: &D,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        extent: Extent2D,
        format: util::Format,
        mipmaps: MipMaps,
        data: &'a [u8],
    ) -> Result<(Self, DeviceBuffer), TextureError> {
        let ((image, staging), mip_levels) = if let MipMaps::Generate = mipmaps {
            let mip_levels = mip_levels_for(extent);
            (
                DeviceImage::device_local_mipmapped(
                    &allocator,
                    command_buffer,
                    extent,
                    format,
                    mip_levels,
                    data,
                )?,
                mip_levels,
            )
        } else {
            (
                DeviceImage::device_local(&allocator, command_buffer, extent, format, data)?,
                1,
            )
        };

        let ret = Self::from_device_image(device, image, format, mip_levels)?;
        Ok((ret, staging))
    }

    pub fn image_view(&self) -> &ImageView {
        &self.image_view
    }

    pub fn vk_image(&self) -> &vk::Image {
        &self.image.vk_image()
    }

    pub fn vk_sampler(&self) -> &vk::Sampler {
        &self.sampler.vk_sampler()
    }

    pub fn extent(&self) -> Extent2D {
        self.image.extent()
    }

    pub fn format(&self) -> util::Format {
        self.image.format()
    }
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TEXTURE")
    }
}

pub struct TextureStorage<T> {
    storage: Storage<T>,
}

impl<T> TextureStorage<T> {
    pub fn new() -> Self {
        Self {
            storage: Storage::<T>::new(),
        }
    }

    pub fn get(&self, handle: &Handle<T>) -> Option<&T> {
        self.storage.get(handle)
    }

    pub fn get_mut(&mut self, handle: &Handle<T>) -> Option<&mut T> {
        self.storage.get_mut(handle)
    }

    pub fn add(&mut self, t: T) -> Handle<T> {
        self.storage.add(t)
    }

    pub fn cached(&self, _descriptor: &TextureDescriptor) -> Option<Handle<T>> {
        None
    }
}
impl TextureStorage<Async<Texture>> {
    pub fn allocate(&mut self, _desc: &TextureDescriptor) -> Handle<Async<Texture>> {
        self.storage.add(Async::Pending)
    }

    pub fn drain_available(&mut self) -> DrainIterator<'_> {
        self.storage
            .drain_filter(|x: &mut Async<Texture>| std::matches!(x, Async::Available(_)))
    }
}

pub type DrainIterator<'a> =
    resurs::DrainFilter<'a, fn(&mut Async<Texture>) -> bool, Async<Texture>>;

impl<T> Default for TextureStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub type Textures = TextureStorage<Texture>;
use crate::resource::Async;
pub type AsyncTextures = TextureStorage<Async<Texture>>;
