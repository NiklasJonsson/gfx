use super::{
    BufferMutability, BufferType, DeviceBuffer, HostBuffer, IndexBufferType, IndexSize, Uniform,
    UniformBufferType, VertexBufferType,
};
use crate::backend;

use crate::util::{as_byte_slice, ByteBuffer};
use backend::buffer::Buffer;
use backend::command::CommandBuffer;
use backend::{AllocatorHandle, MemoryError};

use crate::raw_vk;
use crate::vertex::{VertexDefinition, VertexFormat};

use std::sync::Arc;

pub struct BufferResult<B> {
    pub buffer: B,
    pub transient: Option<Buffer>,
}

pub trait BufferDescriptor {
    type Buffer;
    fn mutability(&self) -> BufferMutability;
    fn n_elems(&self) -> u32;
    fn elem_size(&self) -> u16;
    fn elem_align(&self, _: &AllocatorHandle) -> u16;
    fn data(&self) -> &[u8];
    fn vk_usage_flags(&self) -> raw_vk::BufferUsageFlags;

    fn enqueue_single(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<Self::Buffer>, MemoryError>;

    fn enqueue(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<
        (
            BufferResult<Self::Buffer>,
            Option<BufferResult<Self::Buffer>>,
        ),
        MemoryError,
    > {
        let buf0 = self.enqueue_single(allocator, command_buffer)?;

        let buf1 = if let BufferMutability::Mutable = self.mutability() {
            Some(self.enqueue_single(allocator, command_buffer)?)
        } else {
            None
        };

        Ok((buf0, buf1))
    }
}

#[derive(Clone, Debug)]
enum DescriptorData<'a> {
    Owned(Arc<ByteBuffer>),
    Borrowed(&'a [u8]),
}

pub struct BufferDescriptor2<'a, BT> {
    data: DescriptorData<'a>,
    mutability: BufferMutability,
    elem_size: u16,
    n_elems: u32,
    buffer_type: BT,
}

#[derive(Clone, Debug)]
pub struct OwningBufferDescriptor<BT> {
    data: Arc<ByteBuffer>,
    mutability: BufferMutability,
    elem_size: u16,
    n_elems: u32,
    buffer_type: BT,
}

impl<BT: Clone> OwningBufferDescriptor<BT> {
    pub fn from_host_buffer(hb: &HostBuffer<BT>, mutability: BufferMutability) -> Self {
        Self {
            data: hb.data.clone(),
            mutability,
            elem_size: hb.elem_size,
            n_elems: hb.n_elems,
            buffer_type: hb.buffer_type.clone(),
        }
    }
}

impl<BT: BufferType + Clone> BufferDescriptor for OwningBufferDescriptor<BT> {
    type Buffer = DeviceBuffer<BT>;
    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn vk_usage_flags(&self) -> raw_vk::BufferUsageFlags {
        BT::USAGE
    }

    fn elem_size(&self) -> u16 {
        self.elem_size
    }

    fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        self.buffer_type
            .elem_align(allocator)
            .unwrap_or(self.elem_size())
    }

    fn n_elems(&self) -> u32 {
        self.n_elems
    }

    fn data(&self) -> &[u8] {
        &self.data
    }

    fn enqueue_single(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<Self::Buffer>, MemoryError> {
        let (buffer, transient) =
            Self::Buffer::create(allocator, command_buffer, self, self.buffer_type.clone())?;

        Ok(BufferResult { buffer, transient })
    }
}

pub struct BorrowingBufferDescriptor<'a, BT> {
    data: &'a [u8],
    mutability: BufferMutability,
    elem_size: u16,
    #[allow(dead_code)]
    n_elems: u32,
    buffer_type: BT,
}

impl<'a, BT: BufferType + Clone> BufferDescriptor for BorrowingBufferDescriptor<'a, BT> {
    type Buffer = DeviceBuffer<BT>;

    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn elem_size(&self) -> u16 {
        self.elem_size
    }

    fn n_elems(&self) -> u32 {
        assert_eq!(self.data.len() % self.elem_size() as usize, 0);
        (self.data.len() / self.elem_size() as usize) as u32
    }

    fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        self.buffer_type
            .elem_align(allocator)
            .unwrap_or(self.elem_size())
    }

    fn vk_usage_flags(&self) -> raw_vk::BufferUsageFlags {
        BT::USAGE
    }

    fn data(&self) -> &[u8] {
        self.data
    }

    fn enqueue_single(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<Self::Buffer>, MemoryError> {
        let (buffer, transient) =
            Self::Buffer::create(allocator, command_buffer, self, self.buffer_type.clone())?;

        Ok(BufferResult { buffer, transient })
    }
}

pub type OwningUniformBufferDescriptor = OwningBufferDescriptor<UniformBufferType>;
impl OwningUniformBufferDescriptor {
    pub fn from_vec<T: Copy + Uniform + 'static>(
        data: Vec<T>,
        mutability: BufferMutability,
    ) -> Self {
        let n_elems = data.len() as u32;
        let data = unsafe { Arc::new(ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: std::mem::size_of::<T>() as u16,
            mutability,
            buffer_type: UniformBufferType,
        }
    }
}

pub type OwningVertexBufferDescriptor = OwningBufferDescriptor<VertexBufferType>;
impl OwningVertexBufferDescriptor {
    pub fn from_vec<V: VertexDefinition + Copy + 'static>(
        data: Vec<V>,
        mutability: BufferMutability,
    ) -> Self {
        let n_elems = data.len() as u32;
        let data = unsafe { Arc::new(ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: std::mem::size_of::<V>() as u16,
            mutability,
            buffer_type: VertexBufferType {
                format: V::format(),
            },
        }
    }

    // TODO: unsafe
    pub unsafe fn from_raw(
        data: Vec<u8>,
        format: VertexFormat,
        mutability: BufferMutability,
    ) -> Self {
        assert_eq!(data.len() as u32 % format.size(), 0);
        let n_elems = data.len() as u32 / format.size();
        let data = Arc::new(ByteBuffer::from_vec(data));
        Self {
            data,
            n_elems,
            elem_size: format.size() as u16,
            mutability,
            buffer_type: VertexBufferType { format },
        }
    }
}

pub type OwningIndexBufferDescriptor = OwningBufferDescriptor<IndexBufferType>;
impl OwningIndexBufferDescriptor {
    // TODO: Custom trait here?
    pub fn from_vec<T: num_traits::PrimInt + 'static>(
        data: Vec<T>,
        mutability: BufferMutability,
    ) -> Self {
        let index_size = match std::mem::size_of::<T>() {
            4 => IndexSize::Size32,
            2 => IndexSize::Size16,
            _ => unreachable!("Invalid index type, needs to be either 16 or 32 bits"),
        };
        let n_elems = data.len() as u32;
        let data = unsafe { Arc::new(ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: std::mem::size_of::<T>() as u16,
            mutability,
            buffer_type: IndexBufferType { index_size },
        }
    }
}

pub type BorrowingVertexBufferDescriptor<'a> = BorrowingBufferDescriptor<'a, VertexBufferType>;
impl<'a> BorrowingVertexBufferDescriptor<'a> {
    pub fn from_slice<V: VertexDefinition + Copy>(
        slice: &'a [V],
        mutability: BufferMutability,
    ) -> Self {
        let data = as_byte_slice(slice);
        let format = V::format();
        Self {
            data,
            mutability,
            n_elems: slice.len() as u32,
            elem_size: format.size() as u16,
            buffer_type: VertexBufferType { format },
        }
    }

    pub fn from_raw(data: &'a [u8], format: VertexFormat, mutability: BufferMutability) -> Self {
        assert_eq!(data.len() % format.size() as usize, 0);
        Self {
            data,
            mutability,
            n_elems: data.len() as u32 / format.size(),
            elem_size: format.size() as u16,
            buffer_type: VertexBufferType { format },
        }
    }
}

pub type BorrowingIndexBufferDescriptor<'a> = BorrowingBufferDescriptor<'a, IndexBufferType>;
impl<'a> BorrowingIndexBufferDescriptor<'a> {
    pub fn from_slice<T: num_traits::PrimInt>(
        slice: &'a [T],
        mutability: BufferMutability,
    ) -> Self {
        let n_elems = slice.len() as u32;
        let data = as_byte_slice(slice);
        let index_size = match std::mem::size_of::<T>() {
            4 => IndexSize::Size32,
            2 => IndexSize::Size16,
            _ => unreachable!("Invalid index type, needs to be either 16 or 32 bits"),
        };
        let elem_size = std::mem::size_of::<T>() as u16;

        Self {
            data,
            mutability,
            n_elems,
            elem_size,
            buffer_type: IndexBufferType { index_size },
        }
    }

    pub fn from_raw<T>(
        data: &'a [u8],
        index_size: IndexSize,
        mutability: BufferMutability,
    ) -> Self {
        let elem_size = index_size.size() as usize;
        assert_eq!(data.len() % elem_size, 0);
        Self {
            data,
            mutability,
            n_elems: data.len() as u32 / elem_size as u32,
            elem_size: elem_size as u16,
            buffer_type: IndexBufferType { index_size },
        }
    }
}
