use ash::vk;

use crate::command::CommandPool;
use crate::device::Device;
use crate::mem::DeviceBuffer;
use crate::mem::MemoryError;
use crate::queue::Queue;
use crate::resource::{BufferedStorage, Handle};

use crate::common::MAX_FRAMES_IN_FLIGHT;

use crate::util;

pub enum UniformBufferDescriptor<'a, T> {
    Immutable { data: &'a [T] },
    Uninitialized { n_elems: u64 },
}

pub struct UniformBuffer {
    buffer: DeviceBuffer,
    elem_size: u64,
    stride: u64,
    n_elems: u64,
}

impl UniformBuffer {
    pub fn create<'a, T>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &UniformBufferDescriptor<'a, T>,
    ) -> Result<Self, MemoryError> {
        let elem_size = std::mem::size_of::<T>() as u64;
        let stride = device.uniform_buffer_offset_alignment();
        let (buffer, n_elems) = match descriptor {
            UniformBufferDescriptor::Immutable { data } => (
                DeviceBuffer::device_local_by_staging(
                    device,
                    queue,
                    command_pool,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    util::as_byte_slice(data),
                    elem_size as usize,
                    stride as usize,
                )?,
                data.len() as u64,
            ),
            UniformBufferDescriptor::Uninitialized { n_elems } => (
                DeviceBuffer::empty(
                    device,
                    (elem_size * n_elems) as usize,
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

    pub fn stride(&self) -> u64 {
        self.stride
    }

    pub fn elem_size(&self) -> u64 {
        self.elem_size
    }

    pub fn n_elems(&self) -> u64 {
        self.n_elems
    }

    pub fn size(&self) -> u64 {
        assert!(self.elem_size <= self.stride);
        self.stride * self.n_elems
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
