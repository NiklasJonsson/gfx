use ash::vk;

use crate::backend::vk::{
    command::CommandBuffer, device::AllocatorHandle, util::stride, MemoryError,
};
use vk_mem::{Allocation, AllocationCreateInfo, MemoryUsage};

pub struct Buffer {
    allocator: AllocatorHandle,
    vk_buffer: vk::Buffer,
    allocation: Allocation,
    size: usize,
    is_mapped: bool,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let allocation_info = self.allocator.get_allocation_info(&self.allocation);
        f.debug_struct("Buffer")
            .field("vk_buffer", &self.vk_buffer)
            .field("allocation", &self.allocation)
            .field("size", &self.size)
            .field("allocation_info", &allocation_info)
            .finish()
    }
}

impl Buffer {
    pub fn empty(
        allocator: &AllocatorHandle,
        size: usize,
        buffer_usage_flags: vk::BufferUsageFlags,
        mem_usage: MemoryUsage,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating empty DeviceBuffer with:");
        log::trace!("\tsize: {}", size);
        log::trace!("\tusage: {:?}", buffer_usage_flags);
        log::trace!("\tmemory usage: {:?}", mem_usage);
        assert!(size > 0);

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

        let (vk_buffer, allocation, allocation_info) = allocator
            .create_buffer(&buffer_info, &allocation_create_info)
            .map_err(MemoryError::BufferCreation)?;
        log::trace!("Allocation succeeded: {:?}", &allocation_info);
        log::trace!("Created buffer: {:?}", vk_buffer);

        Ok(Self {
            allocator: AllocatorHandle::clone(allocator),
            vk_buffer,
            allocation,
            size,
            is_mapped: false,
        })
    }

    fn with_data(
        allocator: &AllocatorHandle,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
        buffer_usage: vk::BufferUsageFlags,
        mem_usage: MemoryUsage,
        do_unmap: bool,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating device buffer with data");
        assert!(data.len() % (elem_size as usize) == 0);
        let n_elems = data.len() / (elem_size as usize);
        log::trace!(
            "n_elems: {}, elem_size: {}, elem_align: {}",
            n_elems,
            elem_size,
            elem_align,
        );
        let stride = stride(elem_size, elem_align);
        let size = stride as usize * n_elems;
        log::trace!("Total buffer size: {}", size);

        let mut buffer = Buffer::empty(allocator, size, buffer_usage, mem_usage)?;
        let dst = buffer.map()?;
        let src = data.as_ptr() as *const u8;
        if elem_size == elem_align {
            log::trace!("Straight copy from {:?} to {:?}, size: {}", src, dst, size);
            unsafe {
                std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
            }
        } else {
            log::trace!("Strided copy from {:?} to {:?}, size: {}", src, dst, size);
            for i in 0..n_elems {
                unsafe {
                    let src = src.add(i * (elem_size as usize));
                    let dst = dst.add(i * (stride as usize));
                    std::ptr::copy_nonoverlapping::<u8>(src, dst, elem_size as usize);
                }
            }
        }

        if do_unmap {
            buffer.unmap();
        }

        Ok(buffer)
    }

    pub fn staging_with_data(
        allocator: &AllocatorHandle,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating staging buffer");
        Self::with_data(
            allocator,
            data,
            elem_size,
            elem_align,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::CpuOnly,
            true,
        )
    }

    pub fn persistent_mapped(
        allocator: &AllocatorHandle,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating persistent mapped buffer");
        Self::with_data(
            allocator,
            data,
            elem_size,
            elem_align,
            usage,
            MemoryUsage::CpuToGpu,
            false,
        )
    }

    pub fn device_local(
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
    ) -> Result<(Self, Self), MemoryError> {
        log::trace!("Creating device local buffer (with data from staging)");
        let staging = Self::staging_with_data(allocator, data, elem_size, elem_align)?;

        let dst_buffer = Self::empty(
            allocator,
            staging.size(),
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryUsage::GpuOnly,
        )?;

        command_buffer.copy_buffer(staging.vk_buffer(), dst_buffer.vk_buffer(), staging.size());

        Ok((dst_buffer, staging))
    }
}

impl Buffer {
    pub fn map(&mut self) -> Result<*mut u8, MemoryError> {
        self.is_mapped = true;
        self.allocator
            .map_memory(&self.allocation)
            .map_err(MemoryError::MemoryMapping)
    }

    pub fn unmap(&mut self) {
        assert!(self.is_mapped);
        self.is_mapped = false;
        self.allocator.unmap_memory(&self.allocation)
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.vk_buffer
    }

    pub fn update_data_at(&mut self, data: &[u8], offset: usize) -> Result<(), MemoryError> {
        assert!(self.is_mapped);
        let size = data.len();
        let dst_base = self
            .allocator
            .get_allocation_info(&self.allocation)
            .map_err(MemoryError::MemoryMapping)?
            .get_mapped_data();

        let src = data.as_ptr() as *const u8;
        unsafe {
            assert!(offset + size <= self.size());
            let dst = dst_base.add(offset);
            std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
        }

        Ok(())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl std::ops::Drop for Buffer {
    fn drop(&mut self) {
        if self.is_mapped {
            self.unmap();
        }
        self.allocator
            .destroy_buffer(self.vk_buffer, &self.allocation);
    }
}
