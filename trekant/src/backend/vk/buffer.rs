use ash::vk;

use crate::backend::vk::{
    command::CommandBuffer, device::AllocatorHandle, util::stride, MemoryError,
};
use vma::{Alloc as _, Allocation, AllocationCreateInfo, MemoryUsage};

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
            .field("allocation_info", &allocation_info)
            .field("size", &self.size)
            .field("is_mapped", &self.is_mapped)
            .finish()
    }
}

enum Mappability {
    Mappable,
    NonMappable,
}

/// Copy the elements in src to dst, assmuming:
/// * Elements in src are laid out linearly, with no padding.
/// * Elements in dst are aligned to dst_elem_align, with possible padding in-between. This padding is not touched.
pub unsafe fn element_copy(src: &[u8], dst: *mut u8, elem_size: u16, dst_elem_align: u16) {
    let size = src.len();
    let src = src.as_ptr() as *const u8;
    if elem_size == dst_elem_align {
        log::trace!("Straight copy from {:?} to {:?}, size: {}", src, dst, size);
        unsafe {
            std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
        }
    } else {
        log::trace!("Strided copy from {:?} to {:?}, size: {}", src, dst, size);
        let n_elems = size / elem_size as usize;
        let stride = stride(elem_size, dst_elem_align);
        for i in 0..n_elems {
            unsafe {
                let src = src.add(i * (elem_size as usize));
                let dst = dst.add(i * (stride as usize));
                std::ptr::copy_nonoverlapping::<u8>(src, dst, elem_size as usize);
            }
        }
    }
}

impl Buffer {
    fn empty(
        allocator: &AllocatorHandle,
        size: usize,
        buffer_usage_flags: vk::BufferUsageFlags,
        map: Mappability,
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

        let flags = match map {
            Mappability::Mappable => {
                vma::AllocationCreateFlags::HOST_ACCESS_RANDOM | vma::AllocationCreateFlags::MAPPED
            }
            Mappability::NonMappable => vma::AllocationCreateFlags::empty(),
        };

        let allocation_create_info = AllocationCreateInfo {
            flags,
            usage: mem_usage,
            ..Default::default()
        };

        let (vk_buffer, allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &allocation_create_info) }
                .map_err(MemoryError::BufferCreation)?;
        let allocation_info = allocator.get_allocation_info(&allocation);
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

        let mut buffer = Buffer::empty(
            allocator,
            size,
            buffer_usage,
            Mappability::Mappable,
            mem_usage,
        )?;
        let dst = buffer.map()?;
        unsafe {
            element_copy(data, dst, elem_size, elem_align);
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
            MemoryUsage::AutoPreferHost,
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
            MemoryUsage::Auto,
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
            Mappability::NonMappable,
            MemoryUsage::AutoPreferDevice,
        )?;

        command_buffer.copy_buffer(staging.vk_buffer(), dst_buffer.vk_buffer(), staging.size());

        Ok((dst_buffer, staging))
    }
}

impl Buffer {
    pub fn map(&mut self) -> Result<*mut u8, MemoryError> {
        self.is_mapped = true;
        unsafe { self.allocator.map_memory(&mut self.allocation) }
            .map_err(MemoryError::MemoryMapping)
    }

    pub fn unmap(&mut self) {
        assert!(self.is_mapped);
        self.is_mapped = false;
        unsafe { self.allocator.unmap_memory(&mut self.allocation) }
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.vk_buffer
    }

    /// Write 'data' to the buffer at 'offset', assuming elem_size and elem_align requirements.
    /// Safet
    pub unsafe fn write(
        &mut self,
        offset: usize,
        data: &[u8],
        elem_size: u16,
        dst_elem_align: u16,
    ) {
        assert!(self.is_mapped);
        let src_size = data.len();
        let dst = self
            .allocator
            .get_allocation_info(&self.allocation)
            .mapped_data as *mut u8;
        let dst = dst.add(offset);
        element_copy(data, dst, elem_size, dst_elem_align);
    }

    /// Update the data at `offset` in the buffer from the slice.
    /// SAFETY: The buffer needs to be writable for (offset, offset+size],
    /// e.g. there can be no padding/alignment requirements on the elements
    /// of the buffer.
    pub unsafe fn update_data_at(&mut self, data: &[u8], offset: usize) {
        todo!("Is this needed?");
        self.write(data, offset, data.len(), data.len());
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
        unsafe {
            self.allocator
                .destroy_buffer(self.vk_buffer, &mut self.allocation);
        }
    }
}
