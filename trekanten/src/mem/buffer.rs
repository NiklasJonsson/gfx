use ash::vk;

use vk_mem::{Allocation, AllocationCreateInfo, AllocationInfo, MemoryUsage};

use super::MemoryError;

use crate::command::CommandPool;
use crate::device::AllocatorHandle;
use crate::device::Device;
use crate::queue::Queue;
use crate::resource::Handle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferMutability {
    Immutable,
    Mutable,
}

pub trait BufferDescriptor {
    type Buffer: Buffer;
    fn mutability(&self) -> BufferMutability;
    fn n_elems(&self) -> u32;
    fn elem_size(&self) -> u16;

    fn create(
        &self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
    ) -> Result<Self::Buffer, MemoryError>;
}

pub trait Buffer {
    fn stride(&self) -> u16;
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle<T> {
    h: Handle<T>,
    mutability: BufferMutability,
    idx: u32,
    n_elems: u32,
    elem_size: u16,
    stride: u16,
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
            mutability: BufferMutability::Immutable,
            idx: 0,
            n_elems: 0,
            stride: 0,
            elem_size: 0,
        }
    }
}

impl<T> BufferHandle<T> {
    pub fn sub_buffer(h: Self, idx: u32, n_elems: u32) -> Self {
        assert!((idx + n_elems) <= (h.idx + h.n_elems));
        Self { idx, n_elems, ..h }
    }

    pub unsafe fn from_buffer(
        h: Handle<T>,
        idx: u32,
        n_elems: u32,
        elem_size: u16,
        stride: u16,
        mutability: BufferMutability,
    ) -> Self {
        Self {
            h,
            mutability,
            idx,
            n_elems,
            elem_size,
            stride,
        }
    }

    pub fn handle(&self) -> &Handle<T> {
        &self.h
    }

    pub fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    pub fn split(&self) -> Vec<Self> {
        (0..self.n_elems)
            .map(|i| Self {
                idx: self.idx + i,
                n_elems: 1,
                ..*self
            })
            .collect::<Vec<_>>()
    }

    pub fn offset(&self) -> u64 {
        self.idx as u64 * self.stride as u64
    }

    pub fn size(&self) -> u64 {
        let s = if self.n_elems > 1 {
            (self.n_elems - 1) as u64 * self.stride as u64
        } else {
            0
        };
        s + self.elem_size as u64
    }

    pub fn is_empty(&self) -> bool {
        self.n_elems == 0
    }

    pub fn idx(&self) -> u32 {
        self.idx
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }
}

fn get_aligned_size(n_bytes: usize, elem_size: usize, stride: usize) -> usize {
    let n_elems = n_bytes / elem_size;
    if stride == elem_size {
        n_bytes
    } else {
        assert!(elem_size <= stride);
        n_elems * stride
    }
}

pub struct DeviceBuffer {
    allocator: AllocatorHandle,
    vk_buffer: vk::Buffer,
    allocation: Allocation,
    size: usize,
    allcation_info: AllocationInfo,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("vk_buffer", &self.vk_buffer)
            .field("allocation", &self.allocation)
            .field("size", &self.size)
            .field("allcation_info", &self.allcation_info)
            .finish()
    }
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

        let (vk_buffer, allocation, allcation_info) = allocator
            .create_buffer(&buffer_info, &allocation_create_info)
            .map_err(MemoryError::BufferCreation)?;
        log::trace!("Allocation succeeded: {:?}", &allcation_info);
        log::trace!("Created buffer: {:?}", vk_buffer);

        Ok(Self {
            allocator,
            vk_buffer,
            allocation,
            allcation_info,
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

    fn with_data(
        device: &Device,
        data: &[u8],
        elem_size: usize,
        stride: usize,
        buffer_usage: vk::BufferUsageFlags,
        mem_usage: MemoryUsage,
        do_unmap: bool,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating device buffer with data");
        assert!(data.len() % elem_size == 0);
        let n_elems = data.len() / elem_size;
        log::trace!(
            "n_elems: {}, elem_size: {}, stride: {}",
            n_elems,
            elem_size,
            stride
        );
        let allocator = device.allocator();
        let size = get_aligned_size(data.len(), elem_size, stride);
        log::trace!("Total buffer size: {}", size);

        let staging = DeviceBuffer::empty(device, size, buffer_usage, mem_usage)?;

        let dst = allocator
            .map_memory(&staging.allocation)
            .map_err(MemoryError::MemoryMapping)?;
        let src = data.as_ptr() as *const u8;
        if elem_size == stride {
            log::trace!("Straight copy from {:?} to {:?}, size: {}", src, dst, size);
            unsafe {
                std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
            }
        } else {
            log::trace!("Strided copy from {:?} to {:?}, size: {}", src, dst, size);
            for i in 0..n_elems {
                unsafe {
                    let src = src.add(i * elem_size);
                    let dst = dst.add(i * stride);
                    std::ptr::copy_nonoverlapping::<u8>(src, dst, elem_size);
                }
            }
        }

        if do_unmap {
            allocator
                .unmap_memory(&staging.allocation)
                .map_err(MemoryError::MemoryMapping)?;
        }

        Ok(staging)
    }

    pub fn staging_with_data(
        device: &Device,
        data: &[u8],
        elem_size: usize,
        stride: usize,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating staging buffer");
        Self::with_data(
            device,
            data,
            elem_size,
            stride,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::CpuOnly,
            true,
        )
    }

    pub fn persistent_mapped(
        device: &Device,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        elem_size: usize,
        stride: usize,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating persistent mapped buffer");
        Self::with_data(
            device,
            data,
            elem_size,
            stride,
            usage,
            MemoryUsage::CpuToGpu,
            false,
        )
    }

    pub fn device_local(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        elem_size: usize,
        stride: usize,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating device local buffer (with data from staging)");
        let staging = Self::staging_with_data(device, data, elem_size, stride)?;

        let dst_buffer = Self::empty(
            device,
            staging.size(),
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryUsage::GpuOnly,
        )?;

        let mut cmd_buf = command_pool.begin_single_submit()?;

        cmd_buf
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
