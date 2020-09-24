use ash::vk;

use crate::command::CommandPool;
use crate::device::Device;
use crate::mem;
use crate::mem::{Buffer, BufferDescriptor, BufferHandle};
use crate::queue::Queue;
use crate::util::as_byte_slice;
use crate::vertex::VertexDefinition;
use crate::vertex::VertexFormat;

pub use crate::mem::BufferMutability;

pub struct Mesh {
    pub vertex_buffer: BufferHandle<VertexBuffer>,
    pub index_buffer: BufferHandle<IndexBuffer>,
}

#[derive(Debug, Copy, Clone)]
pub enum IndexSize {
    Size32,
    Size16,
}

impl From<IndexSize> for vk::IndexType {
    fn from(is: IndexSize) -> Self {
        match is {
            IndexSize::Size16 => vk::IndexType::UINT16,
            IndexSize::Size32 => vk::IndexType::UINT32,
        }
    }
}

impl IndexSize {
    fn size(&self) -> u8 {
        match self {
            Self::Size32 => 4,
            Self::Size16 => 2,
        }
    }
}

pub struct IndexBufferDescriptor<'a> {
    data: &'a [u8],
    index_size: IndexSize,
    mutability: BufferMutability,
}

impl<'a> IndexBufferDescriptor<'a> {
    pub fn from_slice<T>(slice: &'a [T], mutability: BufferMutability) -> Self {
        let data = as_byte_slice(slice);
        let index_size = match std::mem::size_of::<T>() {
            4 => IndexSize::Size32,
            2 => IndexSize::Size16,
            _ => unreachable!("Invalid index type, needs to be either 16 or 32 bits"),
        };

        Self {
            data,
            index_size,
            mutability,
        }
    }
}

impl<'a> BufferDescriptor for IndexBufferDescriptor<'a> {
    type Buffer = IndexBuffer;

    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn elem_size(&self) -> u16 {
        self.index_size.size() as u16
    }

    fn n_elems(&self) -> u32 {
        assert_eq!(self.data.len() % self.elem_size() as usize, 0);
        (self.data.len() / self.elem_size() as usize) as u32
    }

    fn create(
        &self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
    ) -> Result<Self::Buffer, mem::MemoryError> {
        Self::Buffer::create(device, queue, command_pool, &self)
    }
}


// TODO?: Merge index, vertex & uniform buffers into one Buffer<Ty>. Ty holds variant info. impl<Concrete> for special functions

#[derive(Debug)]
pub struct IndexBuffer {
    buffer: mem::DeviceBuffer,
    index_size: IndexSize,
}

impl Buffer for IndexBuffer {
    fn stride(&self) -> u16 {
        self.index_size.size() as u16
    }
}

impl IndexBuffer {
    pub fn create<'a>(
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
        descriptor: &IndexBufferDescriptor<'a>,
    ) -> Result<Self, mem::MemoryError> {
        let size = match descriptor.index_size {
            IndexSize::Size16 => 2,
            IndexSize::Size32 => 4,
        };
        log::trace!("Creating index buffer");
        let buffer = match descriptor.mutability() {
            BufferMutability::Immutable => mem::DeviceBuffer::device_local(
                device,
                queue,
                command_pool,
                vk::BufferUsageFlags::INDEX_BUFFER,
                descriptor.data,
                size,
                size,
            ),
            BufferMutability::Mutable => mem::DeviceBuffer::persistent_mapped(
                device,
                vk::BufferUsageFlags::INDEX_BUFFER,
                descriptor.data,
                size,
                size,
            ),
        }?;

        Ok(Self {
            buffer,
            index_size: descriptor.index_size,
        })
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }

    pub fn vk_index_type(&self) -> vk::IndexType {
        vk::IndexType::from(self.index_size)
    }
}

pub struct VertexBufferDescriptor<'a> {
    data: &'a [u8],
    format: VertexFormat,
    mutability: BufferMutability,
}

impl<'a> VertexBufferDescriptor<'a> {
    pub fn from_slice<V: VertexDefinition>(slice: &'a [V], mutability: BufferMutability) -> Self {
        let data = as_byte_slice(slice);
        Self {
            data,
            format: V::format(),
            mutability,
        }
    }

    pub fn from_raw(data: &'a [u8], format: VertexFormat, mutability: BufferMutability) -> Self {
        Self {
            data,
            format,
            mutability,
        }
    }
}

impl<'a> BufferDescriptor for VertexBufferDescriptor<'a> {
    type Buffer = VertexBuffer;
    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn elem_size(&self) -> u16 {
        self.format.size() as u16
    }

    fn n_elems(&self) -> u32 {
        assert_eq!(self.data.len() % self.elem_size() as usize, 0);
        (self.data.len() / self.elem_size() as usize) as u32
    }

    fn create(
        &self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
    ) -> Result<Self::Buffer, mem::MemoryError> {
        Self::Buffer::create(device, queue, command_pool, &self)
    }
}

#[derive(Debug)]
pub struct VertexBuffer {
    pub buffer: mem::DeviceBuffer,
    pub format: VertexFormat,
}

impl Buffer for VertexBuffer {
    fn stride(&self) -> u16 {
        self.format.size() as u16
    }
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
        let size = descriptor.format.size() as usize;
        let buffer = match descriptor.mutability() {
            BufferMutability::Immutable => mem::DeviceBuffer::device_local(
                device,
                queue,
                command_pool,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                descriptor.data,
                size,
                size,
            ),
            BufferMutability::Mutable => mem::DeviceBuffer::persistent_mapped(
                device,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                descriptor.data,
                size,
                size,
            ),
        }?;

        Ok(Self {
            buffer,
            format: descriptor.format.clone(),
        })
    }

    pub fn vk_buffer(&self) -> &vk::Buffer {
        &self.buffer.vk_buffer()
    }
}

pub type VertexBuffers = mem::DeviceBufferStorage<VertexBuffer>;
pub type IndexBuffers = mem::DeviceBufferStorage<IndexBuffer>;
