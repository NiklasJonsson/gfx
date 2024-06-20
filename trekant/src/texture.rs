use std::path::{Path, PathBuf};

use ash::vk::{self, ImageAspectFlags, ImageUsageFlags};

use thiserror::Error;

use crate::backend::{self};

use crate::descriptor::DescriptorData;
use crate::resource::{Handle, Storage};
use crate::util;
use backend::buffer::Buffer;
use backend::command::CommandBuffer;
use backend::image::{Image, ImageDescriptor, ImageView, ImageViewError};
use backend::{AllocatorHandle, HasVkDevice, MemoryError, VkDeviceHandle};
use util::Extent2D;

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

fn load_image<P: AsRef<Path>>(p: &P) -> Result<image::RgbaImage, image::ImageError> {
    let path = p.as_ref();

    log::trace!("Trying to load image from {}", path.display());
    let image = image::open(path)?.to_rgba8();

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureUsage {
    Depth,
    Color,
}

impl From<TextureUsage> for vk::ImageUsageFlags {
    fn from(o: TextureUsage) -> vk::ImageUsageFlags {
        if o == TextureUsage::Color {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TextureType {
    Tex2D,
    // Tex3D,
    TexCube,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureInfo {
    pub extent: Extent2D,
    pub format: util::Format,
    pub usage: TextureUsage,
    pub sampler: SamplerDescriptor,
    pub ty: TextureType,
}

impl Default for TextureInfo {
    fn default() -> Self {
        Self {
            extent: Extent2D {
                width: 0,
                height: 0,
            },
            format: util::Format::RGBA_SRGB,
            usage: TextureUsage::Color,
            sampler: SamplerDescriptor::default(),
            ty: TextureType::Tex2D,
        }
    }
}

/// A texture descriptor is used to describe a texture that is to be created.
#[derive(Debug, Clone)]
pub enum TextureDescriptor<'a> {
    File {
        path: PathBuf,
        format: util::Format,
        mipmaps: MipMaps,
        ty: TextureType,
    },
    Raw {
        data: DescriptorData<'a>,
        extent: Extent2D,
        format: util::Format,
        // TODO: Can this be removed?
        mipmaps: MipMaps,
        ty: TextureType,
    },
    Empty {
        extent: Extent2D,
        usage: TextureUsage,
        format: util::Format,
        sampler: SamplerDescriptor,
        ty: TextureType,
    },
}

impl<'a> TextureDescriptor<'a> {
    pub fn mipmaps(&self) -> MipMaps {
        match self {
            Self::File { mipmaps, .. } => *mipmaps,
            Self::Raw { mipmaps, .. } => *mipmaps,
            Self::Empty { .. } => MipMaps::None,
        }
    }

    pub fn from_vec(
        data: Vec<u8>,
        extent: Extent2D,
        format: util::Format,
        mipmaps: MipMaps,
    ) -> Self {
        Self::Raw {
            data: DescriptorData::from_vec(data),
            extent,
            format,
            mipmaps,
            ty: TextureType::Tex2D,
        }
    }

    pub fn is_empty(&self) -> bool {
        std::matches!(self, TextureDescriptor::Empty { .. })
    }
}

static IMG_RECORDER: std::sync::Mutex<Vec<std::path::PathBuf>> = std::sync::Mutex::new(Vec::new());

fn record_image_load(path: &std::path::Path) {
    let mut recorder = IMG_RECORDER.lock().unwrap();
    for p in &*recorder {
        if p == path {
            log::debug!("Image {} was already loaded", path.display());
            return;
        }
    }
    recorder.push(path.to_path_buf());
}

impl<'a> TextureDescriptor<'a> {
    pub(crate) fn split_desc_data(
        self,
    ) -> Result<(TextureInfo, MipMaps, Option<DescriptorData<'a>>), image::ImageError> {
        match self {
            TextureDescriptor::File {
                path,
                format,
                mipmaps,
                ty,
            } => {
                log::debug!("Loading image at {}", path.display());
                record_image_load(&path);
                let image = load_image(&path)?;
                let extent = Extent2D {
                    width: image.width(),
                    height: image.height(),
                };
                let raw_image_data = image.into_raw();
                Ok((
                    TextureInfo {
                        extent,
                        format,
                        usage: TextureUsage::Color,
                        sampler: SamplerDescriptor::default(),
                        ty,
                    },
                    mipmaps,
                    Some(DescriptorData::from_vec(raw_image_data)),
                ))
            }
            TextureDescriptor::Raw {
                data,
                extent,
                format,
                mipmaps,
                ty,
            } => Ok((
                TextureInfo {
                    extent,
                    format,
                    usage: TextureUsage::Color,
                    sampler: SamplerDescriptor::default(),
                    ty,
                },
                mipmaps,
                Some(data),
            )),
            TextureDescriptor::Empty {
                extent,
                usage,
                format,
                sampler,
                ty,
            } => Ok((
                TextureInfo {
                    extent,
                    format,
                    usage,
                    sampler,
                    ty,
                },
                MipMaps::None,
                None,
            )),
        }
    }
}

/// Load a texture from the provided data.
///
/// The format of the data is described in the descriptor.
///
/// NOTE: The queue that the command buffer is submitted to needs to support
/// graphics commands if mipmap generation is set to true.
///
pub(crate) fn load_texture_from_data<D: HasVkDevice>(
    device: &D,
    allocator: &AllocatorHandle,
    command_buffer: &mut CommandBuffer,
    descriptor: TextureInfo,
    data: &[u8],
    mipmaps: MipMaps,
) -> Result<(Texture, Buffer), TextureError> {
    let TextureInfo {
        extent,
        format,
        usage,
        sampler,
        ty,
    } = descriptor;

    if ty != TextureType::Tex2D {
        unimplemented!("Non-2D textures are not supported");
    }

    let generate_mipmaps = mipmaps == MipMaps::Generate;

    let image_usage = vk::ImageUsageFlags::from(usage)
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let mut mip_levels = 1;

    if generate_mipmaps {
        mip_levels = mip_levels_for(extent);
    }

    let array_layers: u32;
    let image_flags: vk::ImageCreateFlags;
    if ty == TextureType::TexCube {
        array_layers = 6;
        image_flags = vk::ImageCreateFlags::CUBE_COMPATIBLE;
    } else {
        array_layers = 1;
        image_flags = vk::ImageCreateFlags::empty();
    };

    let image = Image::empty_2d(
        allocator,
        ImageDescriptor {
            extent,
            format,
            image_usage,
            image_flags,
            mem_usage: vma::MemoryUsage::AutoPreferDevice,
            mip_levels,
            sample_count: vk::SampleCountFlags::TYPE_1,
            array_layers,
        },
    )?;

    // stride & alignment does not matter as long as they are the same.
    let staging =
        Buffer::staging_with_data(allocator, data, 1 /*elem_size*/, 1 /*stride*/)?;

    backend::vk::image::transition_image_layout(
        command_buffer,
        image.vk_image(),
        mip_levels,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    command_buffer.copy_buffer_to_image(staging.vk_buffer(), image.vk_image(), extent);

    // Transitioned to SHADER_READ_ONLY_OPTIMAL during mipmap generation
    if generate_mipmaps {
        backend::vk::image::generate_mipmaps(command_buffer, &image, extent, mip_levels);
    } else {
        backend::vk::image::transition_image_layout(
            command_buffer,
            image.vk_image(),
            mip_levels,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }

    let tex = Texture::from_image(device, image, sampler)?;
    Ok((tex, staging))
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
    descriptor: SamplerDescriptor,
}

impl Sampler {
    pub fn new<D: HasVkDevice>(device: &D, desc: SamplerDescriptor) -> Result<Self, TextureError> {
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
            descriptor: desc,
        })
    }

    pub fn vk_sampler(&self) -> vk::Sampler {
        self.vk_sampler
    }

    pub fn descriptor(&self) -> SamplerDescriptor {
        self.descriptor
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

fn usage_to_aspect(image_usage: ImageUsageFlags) -> ImageAspectFlags {
    if image_usage.contains(ImageUsageFlags::COLOR_ATTACHMENT) {
        ImageAspectFlags::COLOR
    } else {
        ImageAspectFlags::DEPTH
    }
}

pub struct Texture {
    sampler: Sampler,
    image: Image,
    full_image_view: ImageView,
    // If this texture has array_layers, e.g. for a cube map, these point into each layer.
    sub_image_views: Vec<ImageView>,
}

impl Texture {
    /// Create an empty image (with no data)
    ///
    /// The created image will be allocated on the gpu.
    pub(crate) fn empty<D: HasVkDevice>(
        device: &D,
        allocator: &AllocatorHandle,
        descriptor: TextureInfo,
    ) -> Result<Self, TextureError> {
        let TextureInfo {
            extent,
            format,
            usage,
            sampler,
            ty,
        } = descriptor;

        let array_layers: u32;
        let image_flags: vk::ImageCreateFlags;
        if ty == TextureType::TexCube {
            array_layers = 6;
            image_flags = vk::ImageCreateFlags::CUBE_COMPATIBLE;
        } else {
            array_layers = 1;
            image_flags = vk::ImageCreateFlags::empty();
        };

        let image_usage = vk::ImageUsageFlags::from(usage) | vk::ImageUsageFlags::SAMPLED;
        let mip_levels = 1;
        let image = Image::empty_2d(
            allocator,
            ImageDescriptor {
                extent,
                format,
                image_usage,
                image_flags,
                mem_usage: vma::MemoryUsage::AutoPreferDevice,
                mip_levels,
                sample_count: vk::SampleCountFlags::TYPE_1,
                array_layers,
            },
        )?;
        Self::from_image(device, image, sampler)
    }

    pub(crate) fn from_image<D: HasVkDevice>(
        device: &D,
        image: Image,
        sampler_descriptor: SamplerDescriptor,
    ) -> Result<Self, TextureError> {
        let desc = image.descriptor();
        let aspect = usage_to_aspect(desc.image_usage);
        let image_view_type = match desc.array_layers {
            6 => vk::ImageViewType::CUBE,
            1 => vk::ImageViewType::TYPE_2D,
            n => unimplemented!("Can't handle {n} array layers"),
        };

        let full_image_view = ImageView::new(
            device,
            image.vk_image(),
            desc.format,
            aspect,
            desc.mip_levels,
            image_view_type,
            0,
            desc.array_layers,
        )?;

        let sub_image_views = (0..desc.array_layers)
            .map(|i| {
                ImageView::new(
                    device,
                    image.vk_image(),
                    desc.format,
                    aspect,
                    desc.mip_levels,
                    vk::ImageViewType::TYPE_2D,
                    i,
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let sampler = Sampler::new(device, sampler_descriptor)?;

        Ok(Self {
            image,
            sampler,
            full_image_view,
            sub_image_views,
        })
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    pub fn full_image_view(&self) -> &ImageView {
        &self.full_image_view
    }

    pub fn sub_image_views(&self) -> &[ImageView] {
        &self.sub_image_views
    }

    pub fn sub_image_view(&self, idx: usize) -> &ImageView {
        &self.sub_image_views[idx]
    }

    pub fn vk_image(&self) -> vk::Image {
        self.image.vk_image()
    }

    pub fn vk_sampler(&self) -> vk::Sampler {
        self.sampler.vk_sampler()
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
}
impl TextureStorage<Async<Texture>> {
    pub fn allocate(&mut self) -> Handle<Async<Texture>> {
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
