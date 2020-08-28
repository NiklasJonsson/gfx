use ash::vk;

use vk_mem::{Allocation, AllocationCreateInfo, AllocationInfo, MemoryUsage};

use thiserror::Error;

use crate::command::CommandBuffer;
use crate::command::CommandError;
use crate::command::CommandPool;
use crate::device::AllocatorHandle;
use crate::device::Device;
use crate::queue::Queue;
use crate::queue::QueueError;
use crate::resource::Handle;
use crate::util;

// TODO: Buffer handle should store offset and size in terms of elements instead of bytes,
// as this is what is used when passing data to vulkan (for vbuf & ibuf atleast, what about ubuf?)
// TODO: Call this buffer slice instead?
// TODO: Vulkan takes u64 generally
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle<T> {
    h: Handle<T>,
    // Units are bytes for all
    offset: u32,
    size: u32,
    elem_size: u32,
}

// TODO: Fix these
impl<T> Clone for BufferHandle<T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}
impl<T> Copy for BufferHandle<T> {}
impl<T> Default for BufferHandle<T> {
    fn default() -> Self {
        Self {
            h: Handle::<T>::default(),
            offset: 0,
            size: 0,
            elem_size: 0,
        }
    }
}

impl<T> BufferHandle<T> {
    pub fn from_buffer(h: Handle<T>, offset: u32, size: u32, elem_size: u32) -> Self {
        Self {
            h,
            offset,
            size,
            elem_size,
        }
    }

    pub fn from_typed_buffer<V>(h: Handle<T>, idx: u32, len: u32) -> Self {
        let elem_size = std::mem::size_of::<V>() as u32;
        Self {
            h,
            offset: idx * elem_size,
            size: len * elem_size,
            elem_size,
        }
    }

    pub fn handle(&self) -> &Handle<T> {
        &self.h
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn idx(&self) -> u32 {
        assert!(self.offset % self.elem_size == 0);
        self.offset / self.elem_size
    }

    pub fn len(&self) -> u32 {
        assert!(self.size % self.elem_size == 0);
        self.size / self.elem_size
    }
}

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Failed to create buffer {0}")]
    BufferCreation(vk_mem::Error),
    #[error("Failed to create image {0}")]
    ImageCreation(vk_mem::Error),
    #[error("command error during copy {0}")]
    CopyCommand(#[from] CommandError),
    #[error("queue submission failed {0}")]
    CopySubmit(#[from] QueueError),
    #[error("memory mapping failed {0}")]
    MemoryMapping(vk_mem::Error),
}

pub struct DeviceBuffer {
    allocator: AllocatorHandle,
    vk_buffer: vk::Buffer,
    allocation: Allocation,
    size: usize,
    _allocation_info: AllocationInfo,
}

impl DeviceBuffer {
    pub fn empty(
        device: &Device,
        size: usize,
        buffer_usage_flags: vk::BufferUsageFlags,
        mem_usage: MemoryUsage,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating empty DeviceBuffer with:");
        log::trace!("\tsize: {}", size);
        log::trace!("\tusage: {:?}", buffer_usage_flags);
        log::trace!("\tmemory usage: {:?}", mem_usage);
        let buffer_info = vk::BufferCreateInfo {
            size: size as u64,
            usage: buffer_usage_flags,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let allocation_create_info = AllocationCreateInfo {
            usage: mem_usage,
            ..Default::default()
        };
        let allocator = device.allocator();

        let (vk_buffer, allocation, _allocation_info) = allocator
            .create_buffer(&buffer_info, &allocation_create_info)
            .map_err(MemoryError::BufferCreation)?;
        log::trace!("Allocation succeeded: {:?}", &_allocation_info);

        Ok(Self {
            allocator,
            vk_buffer,
            allocation,
            _allocation_info,
            size,
        })
    }

    pub fn staging_empty(device: &Device, size: usize) -> Result<Self, MemoryError> {
        log::trace!("Creating device buffer, empty staging!");
        DeviceBuffer::empty(
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::CpuOnly,
        )
    }

    pub fn staging_with_data(device: &Device, data: &[u8]) -> Result<Self, MemoryError> {
        log::trace!("Creating device buffer, staging with data");
        let allocator = device.allocator();
        let size = data.len();

        let staging = DeviceBuffer::staging_empty(device, size)?;

        let dst = allocator
            .map_memory(&staging.allocation)
            .map_err(MemoryError::MemoryMapping)?;
        unsafe {
            let src = data.as_ptr() as *const u8;
            log::trace!("Copy from {:?} to {:?}, size: {}", src, dst, size);
            std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
        }
        allocator
            .unmap_memory(&staging.allocation)
            .map_err(MemoryError::MemoryMapping)?;

        Ok(staging)
    }

    pub fn device_local_by_staging(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        usage: vk::BufferUsageFlags,
        data: &[u8],
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating device buffer, local by staging");
        let staging = Self::staging_with_data(device, data)?;

        let dst_buffer = Self::empty(
            device,
            staging.size(),
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryUsage::GpuOnly,
        )?;

        let cmd_buf = command_pool
            .begin_single_submit()?
            .copy_buffer(staging.vk_buffer(), dst_buffer.vk_buffer(), staging.size())
            .end()?;

        queue.submit_and_wait(&cmd_buf)?;

        Ok(dst_buffer)
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.vk_buffer
    }

    pub fn update_data_at(&mut self, data: &[u8], offset: usize) -> Result<(), MemoryError> {
        let size = data.len();

        let dst_base = self
            .allocator
            .map_memory(&self.allocation)
            .map_err(MemoryError::MemoryMapping)?;

        let src = data.as_ptr() as *const u8;
        unsafe {
            assert!(offset + size <= self.size());
            let dst = dst_base.add(offset);
            std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
        }

        self.allocator
            .unmap_memory(&self.allocation)
            .map_err(MemoryError::MemoryMapping)?;

        Ok(())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl std::ops::Drop for DeviceBuffer {
    fn drop(&mut self) {
        if let Err(e) = self
            .allocator
            .destroy_buffer(self.vk_buffer, &self.allocation)
        {
            log::error!("Failed to destroy buffer: {}", e);
        }
    }
}

fn transition_image_layout(
    cmd_buf: CommandBuffer,
    vk_image: &vk::Image,
    mip_levels: u32,
    _vk_format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> CommandBuffer {
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
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => unimplemented!("Unimplemented layout transition"),
    };

    let barrier = vk::ImageMemoryBarrier {
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: *vk_image,
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

    cmd_buf.pipeline_barrier(&barrier, src_stage, dst_stage)
}

// TODO: This code depends on vk_image being TRANSfER_DST_OPTIMAL. We should track this together
// with the image.
fn generate_mipmaps(
    mut cmd_buf: CommandBuffer,
    vk_image: &vk::Image,
    extent: &util::Extent2D,
    mip_levels: u32,
) -> CommandBuffer {
    assert!(cmd_buf.is_started());
    let aspect_mask = vk::ImageAspectFlags::COLOR;

    let mut barrier = vk::ImageMemoryBarrier {
        image: *vk_image,
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

        cmd_buf = cmd_buf
            .pipeline_barrier(
                &barrier,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            )
            .blit_image(vk_image, vk_image, &image_blit)
            .pipeline_barrier(
                &transistion_src_barrier,
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
        &last_mip_level_transition,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    )
}

pub struct DeviceImage {
    allocator: AllocatorHandle,
    vk_image: vk::Image,
    allocation: Allocation,
    _allocation_info: AllocationInfo,
}

impl DeviceImage {
    pub fn empty_2d(
        device: &Device,
        extents: util::Extent2D,
        format: util::Format,
        image_usage: vk::ImageUsageFlags,
        mem_usage: MemoryUsage,
        mip_levels: u32,
        sample_count: vk::SampleCountFlags,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating empty 2D DeviceImage with:");
        log::trace!("\textents: {}", extents);
        log::trace!("\tformat: {:?}", format);
        log::trace!("\tusage: {:?}", image_usage);
        log::trace!("\tmemory properties: {:?}", mem_usage);
        log::trace!("\tmip level: {}", mip_levels);
        log::trace!("\tsample count: {:?}", sample_count);
        log::trace!("\timage tiling {:?}", vk::ImageTiling::OPTIMAL);

        let extents3d = util::Extent3D::from_2d(extents, 1);
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extents3d.into())
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format.into())
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(image_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count);

        let allocation_create_info = AllocationCreateInfo {
            usage: mem_usage,
            ..Default::default()
        };
        let allocator = device.allocator();
        let (vk_image, allocation, _allocation_info) = allocator
            .create_image(&info, &allocation_create_info)
            .map_err(MemoryError::ImageCreation)?;

        Ok(Self {
            allocator,
            vk_image,
            allocation,
            _allocation_info,
        })
    }

    /// Create a device local image, generating mipmaps in the process
    pub fn device_local_mipmapped(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        extent: util::Extent2D,
        format: util::Format,
        mip_levels: u32,
        data: &[u8],
    ) -> Result<Self, MemoryError> {
        let staging = DeviceBuffer::staging_with_data(device, data)?;
        // Both src & dst as we use one mip level to create the next
        let usage = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        let dst_image = Self::empty_2d(
            device,
            extent,
            format,
            usage,
            MemoryUsage::GpuOnly,
            mip_levels,
            vk::SampleCountFlags::TYPE_1,
        )?;

        // Transitioned to SHADER_READ_ONLY_OPTIMAL during mipmap generation
        let cmd_buf = command_pool.begin_single_submit()?;

        let cmd_buf = transition_image_layout(
            cmd_buf,
            &dst_image.vk_image,
            mip_levels,
            format.into(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )
        .copy_buffer_to_image(&staging.vk_buffer, dst_image.vk_image(), &extent);

        let cmd_buf = generate_mipmaps(cmd_buf, dst_image.vk_image(), &extent, mip_levels).end()?;

        queue.submit_and_wait(&cmd_buf)?;

        Ok(dst_image)
    }

    pub fn vk_image(&self) -> &vk::Image {
        &self.vk_image
    }
}

impl std::ops::Drop for DeviceImage {
    fn drop(&mut self) {
        if let Err(e) = self
            .allocator
            .destroy_image(self.vk_image, &self.allocation)
        {
            log::error!("Failed to destroy image: {}", e);
        }
    }
}
