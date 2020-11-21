use ash::vk;

use crate::command::CommandBuffer;
use crate::device::Device;
use crate::mem::{
    BufferDescriptor, BufferMutability, DeviceBufferStorage, MemoryError, TypedBuffer,
};
use crate::util;

pub struct UniformBufferDescriptor<'a, T> {
    pub data: &'a [T],
    pub mutability: BufferMutability,
}

impl<'a, T> BufferDescriptor for UniformBufferDescriptor<'a, T> {
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

    fn stride(&self, device: &Device) -> u16 {
        device.uniform_buffer_offset_alignment() as u16
    }

    fn n_elems(&self) -> u32 {
        self.data.len() as u32
    }

    fn data(&self) -> &[u8] {
        util::as_byte_slice(self.data)
    }

    fn create(
        &self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
    ) -> Result<(Self::Buffer, Option<crate::mem::DeviceBuffer>), MemoryError> {
        UniformBuffer::create(
            device,
            command_buffer,
            self,
            UniformBufferType {
                n_elems: self.n_elems(),
            },
        )
    }
}

#[derive(Debug)]
pub struct UniformBufferType {
    n_elems: u32,
}

pub type UniformBuffer = TypedBuffer<UniformBufferType>;

impl UniformBuffer {
    pub fn update_with<T>(&mut self, data: &T, offset: u64) -> Result<(), MemoryError> {
        let raw_data = util::as_bytes(data);
        self.buffer_mut().update_data_at(raw_data, offset as usize)
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
