use ash::vk;

use crate::command::CommandPool;
use crate::device::Device;
use crate::mem::DeviceBuffer;
use crate::mem::MemoryError;
use crate::queue::Queue;
use crate::resource::{BufferedStorage, Handle};

use crate::common::MAX_FRAMES_IN_FLIGHT;

use crate::util;

pub enum UniformBufferDescriptor<'a> {
    Initialized { data: &'a [u8], elem_size: usize },
    Uninitialized { elem_size: usize, n_elems: usize },
}

impl<'a> UniformBufferDescriptor<'a> {
    pub fn from_slice<V>(slice: &'a [V]) -> Self {
        let data = util::as_byte_slice(slice);

        Self::Initialized {
            elem_size: std::mem::size_of::<V>(),
            data,
        }
    }

    pub fn uninitialized<V>(n_elems: usize) -> Self {
        Self::Uninitialized {
            elem_size: std::mem::size_of::<V>(),
            n_elems,
        }
    }
}

pub struct UniformBuffer {
    buffer: DeviceBuffer,
    elem_size: usize,
    n_elems: usize,
}

impl UniformBuffer {
    pub fn create<'a>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &UniformBufferDescriptor<'a>,
    ) -> Result<Self, MemoryError> {
        let (buffer, elem_size, n_elems) = match descriptor {
            UniformBufferDescriptor::Initialized { data, elem_size } => (
                DeviceBuffer::device_local_by_staging(
                    device,
                    queue,
                    command_pool,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    data,
                )?,
                *elem_size,
                data.len() / elem_size,
            ),
            UniformBufferDescriptor::Uninitialized { elem_size, n_elems } => (
                DeviceBuffer::empty(
                    device,
                    elem_size * n_elems,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk_mem::MemoryUsage::CpuToGpu,
                )?,
                *elem_size,
                *n_elems,
            ),
        };

        Ok(Self {
            buffer,
            elem_size,
            n_elems,
        })
    }

    pub fn update_with<T>(&mut self, data: &T) -> Result<(), MemoryError> {
        let raw_data = util::as_bytes(data);
        self.buffer.update_data_at(raw_data, 0)
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }

    pub fn elem_size(&self) -> usize {
        self.elem_size
    }

    pub fn n_elems(&self) -> usize {
        self.n_elems
    }

    pub fn size(&self) -> usize {
        self.n_elems * self.elem_size
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

    pub fn create<'a>(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &UniformBufferDescriptor<'a>,
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
