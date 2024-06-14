use ash::vk;

use thiserror::Error;

use crate::{backend, Extent2D};

use backend::command::{CommandBuffer, CommandBufferType};
use backend::device::{HasVkDevice, VkDeviceHandle};
use backend::{AllocatorHandle, MemoryError};
use vma::{Alloc as _, Allocation, AllocationCreateInfo, MemoryUsage};

use crate::util;

#[derive(Debug, Clone, Error)]
pub enum ImageViewError {
    #[error("ImageView creation failed: {0}")]
    Creation(vk::Result),
}

pub struct ImageView {
    vk_image_view: vk::ImageView,
    vk_device: VkDeviceHandle,
}

impl std::fmt::Debug for ImageView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageView")
            .field("image_view", &self.vk_image_view)
            .finish()
    }
}

impl std::ops::Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.vk_device.destroy_image_view(self.vk_image_view, None);
        }
    }
}

impl ImageView {
    // TODO: Refactor? Lots of parameters...
    pub fn new<D: HasVkDevice>(
        device: &D,
        vk_image: vk::Image,
        format: util::Format,
        aspect_mask: vk::ImageAspectFlags,
        mip_levels: u32,
        image_view_type: vk::ImageViewType,
        base_array_layer: u32,
        layer_count: u32,
    ) -> Result<Self, ImageViewError> {
        let vk_format = format.into();
        let comp_mapping = vk::ComponentMapping {
            r: vk::ComponentSwizzle::R,
            g: vk::ComponentSwizzle::G,
            b: vk::ComponentSwizzle::B,
            a: vk::ComponentSwizzle::A,
        };

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer,
            layer_count,
        };

        let info = vk::ImageViewCreateInfo::builder()
            .image(vk_image)
            .view_type(image_view_type)
            .format(vk_format)
            .components(comp_mapping)
            .subresource_range(subresource_range);

        let vk_image_view = unsafe {
            device
                .vk_device()
                .create_image_view(&info, None)
                .map_err(ImageViewError::Creation)?
        };

        log::trace!("Created image view {vk_image_view:?}");

        Ok(Self {
            vk_image_view,
            vk_device: device.vk_device(),
        })
    }

    pub fn vk_image_view(&self) -> vk::ImageView {
        self.vk_image_view
    }
}

pub fn transition_image_layout(
    cmd_buf: &mut CommandBuffer,
    vk_image: vk::Image,
    mip_levels: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    // Note: The barrier below does not really matter at the moment as we wait on the fence
    // directly after submitting. If the code is used elsewhere, it makes the following
    // assumptions:
    // * The image is only read in the fragment shader
    // * The image is not an image array
    // * The image is only used in one queue

    let (src_mask, src_stage, dst_mask, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
            if cmd_buf.ty() == CommandBufferType::Graphics {
                (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::SHADER_READ,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                )
            } else {
                (
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                )
            }
        }
        _ => unimplemented!("Unimplemented layout transition"),
    };

    let barrier = vk::ImageMemoryBarrier {
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: vk_image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_access_mask: src_mask,
        dst_access_mask: dst_mask,
        ..Default::default()
    };

    cmd_buf.pipeline_barrier(&[barrier], src_stage, dst_stage);
}

/// This requires the incoming image to be TRANSFER_DST_OPTIMAL
pub fn generate_mipmaps(
    cmd_buf: &mut CommandBuffer,
    image: &Image,
    extent: util::Extent2D,
    mip_levels: u32,
) {
    log::trace!(
        "Generating mipmaps for {:?}. extent = {}. mip_levels = {}",
        image.vk_image(),
        extent,
        mip_levels
    );
    assert!(cmd_buf.is_started());
    let vk_image = image.vk_image();
    let aspect_mask = vk::ImageAspectFlags::COLOR;

    let mut barrier = vk::ImageMemoryBarrier {
        image: vk_image,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut mip_width = extent.width;
    let mut mip_height = extent.height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        let src_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: mip_width as i32,
                y: mip_height as i32,
                z: 1,
            },
        ];

        let dst_x = if mip_width > 1 { mip_width / 2 } else { 1 } as i32;
        let dst_y = if mip_height > 1 { mip_height / 2 } else { 1 } as i32;
        let dst_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: dst_x,
                y: dst_y,
                z: 1,
            },
        ];

        let image_blit = vk::ImageBlit {
            src_offsets,
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: i - 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets,
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask,
                mip_level: i,
                base_array_layer: 0,
                layer_count: 1,
            },
        };

        let transistion_src_barrier = vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_access_mask: vk::AccessFlags::TRANSFER_READ,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..barrier
        };

        cmd_buf
            .pipeline_barrier(
                &[barrier],
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            )
            .blit_image(vk_image, vk_image, image_blit)
            .pipeline_barrier(
                &[transistion_src_barrier],
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            );

        mip_width = if mip_width > 1 {
            mip_width / 2
        } else {
            mip_width
        };
        mip_height = if mip_height > 1 {
            mip_height / 2
        } else {
            mip_height
        };
    }

    let last_mip_level_transition = vk::ImageMemoryBarrier {
        subresource_range: vk::ImageSubresourceRange {
            base_mip_level: mip_levels - 1,
            ..barrier.subresource_range
        },
        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
        dst_access_mask: vk::AccessFlags::SHADER_READ,
        ..barrier
    };

    cmd_buf.pipeline_barrier(
        &[last_mip_level_transition],
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    );
}

#[derive(Debug, Clone, Copy)]
pub struct ImageDescriptor {
    pub extent: Extent2D,
    pub format: util::Format,
    pub image_usage: vk::ImageUsageFlags,
    pub image_flags: vk::ImageCreateFlags,
    pub mem_usage: MemoryUsage,
    pub mip_levels: u32,
    pub sample_count: vk::SampleCountFlags,
    pub array_layers: u32,
}

pub struct Image {
    allocator: AllocatorHandle,
    vk_image: vk::Image,
    allocation: Allocation,
    descriptor: ImageDescriptor,
}

impl Image {
    pub fn empty_2d(
        allocator: &AllocatorHandle,
        descriptor: ImageDescriptor,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating empty 2D DeviceImage with:");
        log::trace!("{:?}", descriptor);

        let layout = vk::ImageLayout::UNDEFINED;
        let extent3d = util::Extent3D::from_2d(descriptor.extent, 1);
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent3d.into())
            .flags(descriptor.image_flags)
            .mip_levels(descriptor.mip_levels)
            .array_layers(descriptor.array_layers)
            .format(descriptor.format.into())
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(layout)
            .usage(descriptor.image_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(descriptor.sample_count);

        let allocation_create_info = AllocationCreateInfo {
            usage: descriptor.mem_usage,
            ..Default::default()
        };
        let (vk_image, allocation) =
            unsafe { allocator.create_image(&info, &allocation_create_info) }
                .map_err(MemoryError::ImageCreation)?;
        log::trace!("Created image {vk_image:?}");

        Ok(Self {
            allocator: AllocatorHandle::clone(allocator),
            vk_image,
            allocation,
            descriptor,
        })
    }
}

impl Image {
    pub fn vk_image(&self) -> vk::Image {
        self.vk_image
    }

    pub fn extent(&self) -> util::Extent2D {
        self.descriptor.extent
    }

    pub fn format(&self) -> util::Format {
        self.descriptor.format
    }

    pub fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }
}

impl std::ops::Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_image(self.vk_image, &mut self.allocation);
        }
    }
}

// TODO: Remove
/// Utility for holding an image and an image view into it. Only for use inside of the trekant crate
#[allow(dead_code)]
pub(crate) struct ImageAttachment {
    pub image: Image,
    pub image_view: ImageView,
}
