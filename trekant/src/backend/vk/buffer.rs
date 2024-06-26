use ash::vk;

use crate::backend::vk::{
    command::CommandBuffer, device::AllocatorHandle, util::compute_stride, MemoryError,
};
use vma::{Alloc as _, Allocation, AllocationCreateInfo, MemoryUsage};

pub const MIN_BUFFER_OFFSET: u16 = 256;

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
        let stride = compute_stride(elem_size, elem_align);
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
            crate::util::copy_nonoverlapping_aligned(data, dst, elem_size, elem_align);
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
        let ptr = unsafe { self.allocator.map_memory(&mut self.allocation) }
            .map_err(MemoryError::MemoryMapping)?;
        self.is_mapped = true;
        Ok(ptr)
    }

    pub fn unmap(&mut self) {
        assert!(self.is_mapped);
        unsafe { self.allocator.unmap_memory(&mut self.allocation) }
        self.is_mapped = false;
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.vk_buffer
    }

    fn mapped_data(&self) -> *mut std::ffi::c_void {
        assert!(self.is_mapped);
        self.allocator
            .get_allocation_info(&self.allocation)
            .mapped_data
    }

    pub fn mut_ptr(&mut self) -> *mut u8 {
        self.mapped_data() as *mut u8
    }

    pub fn ptr(&mut self) -> *const u8 {
        self.mapped_data() as *const u8
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
