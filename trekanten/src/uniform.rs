use ash::vk;

use crate::command::CommandPool;
use crate::device::Device;
use crate::mem::{Buffer, BufferDescriptor, BufferMutability, DeviceBuffer, MemoryError};
use crate::queue::Queue;
use crate::resource::{BufferedStorage, Handle};

use crate::common::MAX_FRAMES_IN_FLIGHT;

use crate::util;

pub enum UniformBufferDescriptor<'a, T> {
    Immutable { data: &'a [T] },
    Uninitialized { n_elems: u32 },
}

impl<'a, T> BufferDescriptor for UniformBufferDescriptor<'a, T> {
    type Buffer = UniformBuffer;
    fn mutability(&self) -> BufferMutability {
        match self {
            UniformBufferDescriptor::Immutable { .. } => BufferMutability::Immutable,
            UniformBufferDescriptor::Uninitialized { .. } => BufferMutability::Mutable,
        }
    }

    fn elem_size(&self) -> u16 {
        std::mem::size_of::<T>() as u16
    }

    fn n_elems(&self) -> u32 {
        match self {
            UniformBufferDescriptor::Immutable { data } => data.len() as u32,
            UniformBufferDescriptor::Uninitialized { n_elems } => *n_elems,
        }
    }

    fn create(
        &self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
    ) -> Result<Self::Buffer, MemoryError> {
        UniformBuffer::create(device, queue, command_pool, self)
    }
}

pub struct UniformBuffer {
    buffer: DeviceBuffer,
    elem_size: u16,
    stride: u16,
    n_elems: u32,
}

impl Buffer for UniformBuffer {
    fn stride(&self) -> u16 {
        self.stride
    }
}

impl UniformBuffer {
    pub fn create<'a, T>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &UniformBufferDescriptor<'a, T>,
    ) -> Result<Self, MemoryError> {
        let elem_size = std::mem::size_of::<T>() as u16;
        let stride = device.uniform_buffer_offset_alignment() as u16;
        let (buffer, n_elems) = match descriptor {
            // TODO: Immutable doesn't really need to be double buffered
            UniformBufferDescriptor::Immutable { data } => (
                DeviceBuffer::device_local(
                    device,
                    queue,
                    command_pool,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    util::as_byte_slice(data),
                    elem_size as usize,
                    stride as usize,
                )?,
                data.len() as u32,
            ),
            UniformBufferDescriptor::Uninitialized { n_elems } => (
                DeviceBuffer::empty(
                    device,
                    elem_size as usize * *n_elems as usize,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk_mem::MemoryUsage::CpuToGpu,
                )?,
                *n_elems,
            ),
        };

        Ok(Self {
            buffer,
            elem_size,
            stride,
            n_elems,
        })
    }

    pub fn update_with<T>(&mut self, data: &T, offset: u64) -> Result<(), MemoryError> {
        let raw_data = util::as_bytes(data);
        self.buffer.update_data_at(raw_data, offset as usize)
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }

    pub fn stride(&self) -> u16 {
        self.stride
    }

    pub fn elem_size(&self) -> u16 {
        self.elem_size
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }

    pub fn size(&self) -> u64 {
        assert!(self.elem_size <= self.stride);
        self.stride as u64 * self.n_elems as u64
    }
}

impl std::fmt::Debug for UniformBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "UNIFORM BUFFER, elem_size * n_elems = {} * {} = {}",
            self.elem_size(),
            self.n_elems(),
            self.size()
        )
    }
}

#[derive(Default)]
pub struct UniformBuffers {
    storage: BufferedStorage<UniformBuffer>,
}

impl UniformBuffers {
    pub fn new() -> Self {
        Self {
            storage: Default::default(),
        }
    }

    pub fn create<'a, T>(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &UniformBufferDescriptor<'a, T>,
    ) -> Result<Handle<UniformBuffer>, MemoryError> {
        let u_buffer0 = UniformBuffer::create(device, queue, command_pool, descriptor)?;
        let u_buffer1 = UniformBuffer::create(device, queue, command_pool, descriptor)?;
        Ok(self.storage.add([u_buffer0, u_buffer1]))
    }

    pub fn get(&self, h: &Handle<UniformBuffer>, frame_idx: usize) -> Option<&UniformBuffer> {
        self.storage.get(h, frame_idx)
    }

    pub fn get_all(
        &self,
        h: &Handle<UniformBuffer>,
    ) -> Option<&[UniformBuffer; MAX_FRAMES_IN_FLIGHT]> {
        self.storage.get_all(h)
    }

    pub fn get_mut(
        &mut self,
        h: &Handle<UniformBuffer>,
        frame_idx: usize,
    ) -> Option<&mut UniformBuffer> {
        self.storage.get_mut(h, frame_idx)
    }
}
