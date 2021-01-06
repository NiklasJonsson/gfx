use ash::vk;

use crate::command::CommandBuffer;
use crate::device::Device;
use crate::mem::{
    AsyncDeviceBufferStorage, BufferDescriptor, BufferMutability, BufferResult,
    DeviceBufferStorage, MemoryError, TypedBuffer,
};
use crate::util;

use std::sync::Arc;

pub struct UniformBufferDescriptor<'a, T> {
    pub data: &'a [T],
    pub mutability: BufferMutability,
}

impl<'a, T: Copy> BufferDescriptor for UniformBufferDescriptor<'a, T> {
    type Buffer = UniformBuffer;
    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn vk_usage_flags(&self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::UNIFORM_BUFFER
    }

    fn elem_size(&self) -> u16 {
        std::mem::size_of::<T>() as u16
    }

    fn elem_align(&self, device: &Device) -> u16 {
        device.uniform_buffer_offset_alignment() as u16
    }

    fn n_elems(&self) -> u32 {
        self.data.len() as u32
    }

    fn data(&self) -> &[u8] {
        util::as_byte_slice(self.data)
    }

    fn enqueue_single(
        &self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<Self::Buffer>, MemoryError> {
        let (buffer, transient) = UniformBuffer::create(
            device,
            command_buffer,
            self,
            UniformBufferType {
                n_elems: self.n_elems(),
            },
        )?;

        Ok(BufferResult { buffer, transient })
    }
}

#[derive(Debug)]
pub struct UniformBufferType {
    n_elems: u32,
}

pub struct OwningUniformBufferDescriptor {
    pub data: Arc<util::ByteBuffer>,
    pub mutability: BufferMutability,
    elem_size: u16,
    n_elems: u32,
}

impl OwningUniformBufferDescriptor {
    // TODO: Here there should be a trait that ensures std140 layout
    pub fn from_vec<T: Copy>(data: Vec<T>, mutability: BufferMutability) -> Self {
        let n_elems = data.len() as u32;
        let data = unsafe { Arc::new(util::ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: std::mem::size_of::<T>() as u16,
            mutability,
        }
    }
}

impl BufferDescriptor for OwningUniformBufferDescriptor {
    type Buffer = UniformBuffer;
    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn vk_usage_flags(&self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::UNIFORM_BUFFER
    }

    fn elem_size(&self) -> u16 {
        self.elem_size
    }

    fn elem_align(&self, device: &Device) -> u16 {
        device.uniform_buffer_offset_alignment() as u16
    }

    fn n_elems(&self) -> u32 {
        self.n_elems
    }

    fn data(&self) -> &[u8] {
        &self.data
    }

    fn enqueue_single(
        &self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<Self::Buffer>, MemoryError> {
        let (buffer, transient) = UniformBuffer::create(
            device,
            command_buffer,
            self,
            UniformBufferType {
                n_elems: self.n_elems(),
            },
        )?;

        Ok(BufferResult { buffer, transient })
    }
}

pub type UniformBuffer = TypedBuffer<UniformBufferType>;

impl UniformBuffer {
    pub fn update_with<T: Copy>(&mut self, data: &T, idx: u64) -> Result<(), MemoryError> {
        let raw_data = util::as_bytes(data);
        let offset = (idx * self.stride() as u64) as usize;
        self.buffer_mut().update_data_at(raw_data, offset)
    }

    pub fn n_elems(&self) -> u32 {
        self.buffer_type().n_elems
    }

    pub fn size(&self) -> u64 {
        assert!(self.elem_size() <= self.stride());
        self.stride() as u64 * self.n_elems() as u64
    }
}

pub type UniformBuffers = DeviceBufferStorage<UniformBuffer>;
pub type AsyncUniformBuffers = AsyncDeviceBufferStorage<UniformBuffer>;
