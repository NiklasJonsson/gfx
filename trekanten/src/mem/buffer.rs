use ash::vk;

use vk_mem::{Allocation, AllocationCreateInfo, MemoryUsage};

use crate::command::CommandBuffer;
use crate::device::AllocatorHandle;
use crate::device::Device;
use crate::resource::Handle;

use crate::mem::MemoryError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferMutability {
    Immutable,
    Mutable,
}

pub struct BufferResult<B> {
    pub buffer: B,
    pub transient: Option<DeviceBuffer>,
}

pub trait BufferDescriptor {
    type Buffer;
    fn mutability(&self) -> BufferMutability;
    fn n_elems(&self) -> u32;
    fn elem_size(&self) -> u16;
    fn elem_align(&self, _: &Device) -> u16;
    fn data(&self) -> &[u8];
    fn vk_usage_flags(&self) -> vk::BufferUsageFlags;

    fn enqueue_single(
        &self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<Self::Buffer>, MemoryError>;

    fn enqueue(
        &self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
    ) -> Result<
        (
            BufferResult<Self::Buffer>,
            Option<BufferResult<Self::Buffer>>,
        ),
        MemoryError,
    > {
        let buf0 = self.enqueue_single(device, command_buffer)?;

        let buf1 = if let BufferMutability::Mutable = self.mutability() {
            Some(self.enqueue_single(device, command_buffer)?)
        } else {
            None
        };

        Ok((buf0, buf1))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BufferHandle<T> {
    h: Handle<T>,
    mutability: BufferMutability,
    idx: u32,
    n_elems: u32,
}

// TODO: Fix these
impl<T> Clone for BufferHandle<T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}
impl<T> Copy for BufferHandle<T> {}

use crate::resource::Async;

impl<T> BufferHandle<T> {
    pub fn sub_buffer(h: Self, idx: u32, n_elems: u32) -> Self {
        assert!((idx + n_elems) <= (h.idx + h.n_elems));
        Self { idx, n_elems, ..h }
    }

    pub unsafe fn from_buffer(
        h: Handle<T>,
        idx: u32,
        n_elems: u32,
        mutability: BufferMutability,
    ) -> Self {
        Self {
            h,
            mutability,
            idx,
            n_elems,
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

    pub fn is_empty(&self) -> bool {
        self.n_elems == 0
    }

    pub fn idx(&self) -> u32 {
        self.idx
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }

    pub fn wrap_async(&self) -> BufferHandle<Async<T>> {
        BufferHandle::<Async<T>> {
            h: self.h.wrap_async(),
            idx: self.idx,
            n_elems: self.n_elems,
            mutability: self.mutability,
        }
    }
}

pub struct DeviceBuffer {
    allocator: AllocatorHandle,
    vk_buffer: vk::Buffer,
    allocation: Allocation,
    size: usize,
    is_mapped: bool,
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let allocation_info = self.allocator.get_allocation_info(&self.allocation);
        f.debug_struct("DeviceBuffer")
            .field("vk_buffer", &self.vk_buffer)
            .field("allocation", &self.allocation)
            .field("size", &self.size)
            .field("allocation_info", &allocation_info)
            .finish()
    }
}

fn stride(elem_size: u16, elem_align: u16) -> u16 {
    let padding = if elem_size == elem_align { 0 } else { 1 };
    ((elem_size / elem_align) + padding) * elem_align
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

        let (vk_buffer, allocation, allocation_info) = allocator
            .create_buffer(&buffer_info, &allocation_create_info)
            .map_err(MemoryError::BufferCreation)?;
        log::trace!("Allocation succeeded: {:?}", &allocation_info);
        log::trace!("Created buffer: {:?}", vk_buffer);

        Ok(Self {
            allocator,
            vk_buffer,
            allocation,
            size,
            is_mapped: false,
        })
    }

    fn with_data(
        device: &Device,
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

        let mut buffer = DeviceBuffer::empty(device, size, buffer_usage, mem_usage)?;
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
        device: &Device,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating staging buffer");
        Self::with_data(
            device,
            data,
            elem_size,
            elem_align,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::CpuOnly,
            true,
        )
    }

    pub fn persistent_mapped(
        device: &Device,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
    ) -> Result<Self, MemoryError> {
        log::trace!("Creating persistent mapped buffer");
        Self::with_data(
            device,
            data,
            elem_size,
            elem_align,
            usage,
            MemoryUsage::CpuToGpu,
            false,
        )
    }

    pub fn device_local(
        device: &Device,
        command_buffer: &mut CommandBuffer,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        elem_size: u16,
        elem_align: u16,
    ) -> Result<(Self, Self), MemoryError> {
        log::trace!("Creating device local buffer (with data from staging)");
        let staging = Self::staging_with_data(device, data, elem_size, elem_align)?;

        let dst_buffer = Self::empty(
            device,
            staging.size(),
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            MemoryUsage::GpuOnly,
        )?;

        command_buffer.copy_buffer(staging.vk_buffer(), dst_buffer.vk_buffer(), staging.size());

        Ok((dst_buffer, staging))
    }
}

impl DeviceBuffer {
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

impl std::ops::Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.is_mapped {
            self.unmap();
        }
        self.allocator
            .destroy_buffer(self.vk_buffer, &self.allocation);
    }
}

#[derive(Debug)]
pub struct TypedBuffer<BT> {
    buffer: DeviceBuffer,
    elem_size: u16,
    stride: u16,
    buffer_type: BT,
}

impl<BT> TypedBuffer<BT> {
    pub fn create(
        device: &Device,
        command_buffer: &mut CommandBuffer,
        descriptor: &impl BufferDescriptor,
        buffer_type: BT,
    ) -> Result<(Self, Option<DeviceBuffer>), MemoryError> {
        log::trace!("Creating buffer");
        let elem_size = descriptor.elem_size();
        let elem_align = descriptor.elem_align(device);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();

        let (buffer, staging) = match descriptor.mutability() {
            BufferMutability::Immutable => {
                let (buffer, staging) = DeviceBuffer::device_local(
                    device,
                    command_buffer,
                    vk_buffer_usage_flags,
                    data,
                    elem_size,
                    elem_align,
                )?;
                (buffer, Some(staging))
            }
            BufferMutability::Mutable => (
                DeviceBuffer::persistent_mapped(
                    device,
                    vk_buffer_usage_flags,
                    data,
                    elem_size,
                    elem_align,
                )?,
                None,
            ),
        };

        let stride = stride(elem_size, elem_align);

        Ok((
            Self {
                buffer,
                elem_size,
                stride,
                buffer_type,
            },
            staging,
        ))
    }

    pub fn recreate(
        &mut self,
        device: &Device,
        descriptor: &impl BufferDescriptor,
    ) -> Result<(), MemoryError> {
        assert!(descriptor.mutability() == BufferMutability::Mutable);
        let elem_size = descriptor.elem_size();
        let n_elems = descriptor.n_elems();
        let elem_align = descriptor.elem_align(device);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();
        let stride = stride(elem_size, elem_align);
        let size = stride as usize * n_elems as usize;
        if size > self.buffer.size() {
            self.buffer = DeviceBuffer::persistent_mapped(
                device,
                vk_buffer_usage_flags,
                data,
                elem_size,
                stride,
            )?;
        } else {
            self.buffer.update_data_at(descriptor.data(), 0)?;
        }
        self.elem_size = elem_size;
        self.stride = stride;
        Ok(())
    }

    pub fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        &mut self.buffer
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }

    pub fn elem_size(&self) -> u16 {
        self.elem_size
    }

    pub fn buffer_type(&self) -> &BT {
        &self.buffer_type
    }

    pub fn stride(&self) -> u16 {
        self.stride
    }
}
