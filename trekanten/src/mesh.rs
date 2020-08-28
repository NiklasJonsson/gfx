use ash::vk;

use crate::command::CommandPool;
use crate::device::Device;
use crate::mem;
use crate::mem::BufferHandle;
use crate::queue::Queue;
use crate::util::as_byte_slice;
use crate::vertex::VertexDefinition;
use crate::vertex::VertexFormat;

pub struct Mesh {
    pub vertex_buffer: BufferHandle<VertexBuffer>,
    pub index_buffer: BufferHandle<IndexBuffer>,
}

#[derive(Debug, Copy, Clone)]
pub enum IndexSize {
    Size32,
    Size16,
}

pub struct IndexBufferDescriptor<'a> {
    data: &'a [u8],
    index_size: IndexSize,
}

impl<'a> IndexBufferDescriptor<'a> {
    pub fn from_slice<T>(slice: &'a [T]) -> Self {
        let data = as_byte_slice(slice);
        let index_size = match std::mem::size_of::<T>() {
            4 => IndexSize::Size32,
            2 => IndexSize::Size16,
            _ => unreachable!("Invalid index type, needs to be either 16 or 32 bits"),
        };

        Self { data, index_size }
    }
}

pub struct IndexBuffer {
    pub buffer: mem::DeviceBuffer,
    pub index_type: vk::IndexType,
}

impl IndexBuffer {
    pub fn create<'a>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &IndexBufferDescriptor<'a>,
    ) -> Result<Self, mem::MemoryError> {
        log::trace!("Creating index buffer");
        let buffer = mem::DeviceBuffer::device_local_by_staging(
            device,
            queue,
            command_pool,
            vk::BufferUsageFlags::INDEX_BUFFER,
            descriptor.data,
        )?;

        let index_type = match descriptor.index_size {
            IndexSize::Size16 => vk::IndexType::UINT16,
            IndexSize::Size32 => vk::IndexType::UINT32,
        };

        Ok(Self { buffer, index_type })
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }

    pub fn vk_index_type(&self) -> vk::IndexType {
        self.index_type
    }
}

pub struct VertexBufferDescriptor<'a> {
    data: &'a [u8],
    format: VertexFormat,
}

impl<'a> VertexBufferDescriptor<'a> {
    pub fn from_slice<V: VertexDefinition>(slice: &'a [V]) -> Self {
        let data = as_byte_slice(slice);
        Self {
            data,
            format: V::format(),
        }
    }

    pub fn from_raw(data: &'a [u8], format: VertexFormat) -> Self {
        Self { data, format }
    }
}

pub struct VertexBuffer {
    pub buffer: mem::DeviceBuffer,
    pub format: VertexFormat,
}

impl VertexBuffer {
    pub fn create<'a>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &VertexBufferDescriptor<'a>,
    ) -> Result<Self, mem::MemoryError> {
        log::trace!("Creating vertex buffer");
        log::trace!(
            "\tslice: ({:?}, {})",
            descriptor.data.as_ptr(),
            descriptor.data.len()
        );
        log::trace!("\tformat: {:#?}", descriptor.format);
        let buffer = mem::DeviceBuffer::device_local_by_staging(
            device,
            queue,
            command_pool,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            descriptor.data,
        )?;

        Ok(Self {
            buffer,
            format: descriptor.format.clone(),
        })
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }
}
