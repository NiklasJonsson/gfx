mod buffer_storage;

pub use buffer_storage::*;

use ash::vk;

use crate::backend;
use crate::resource::Handle;
use crate::vertex::{VertexDefinition, VertexFormat};
use backend::command::CommandBuffer;
use backend::{buffer::DeviceBuffer, util::stride, AllocatorHandle, MemoryError};

use crate::util::{as_byte_slice, as_bytes, ByteBuffer};

use std::sync::Arc;

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
    fn elem_align(&self, _: &AllocatorHandle) -> u16;
    fn data(&self) -> &[u8];
    fn vk_usage_flags(&self) -> vk::BufferUsageFlags;

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
#[derive(Debug)]
pub struct BufferHandle<T> {
    h: Handle<T>,
    mutability: BufferMutability,
    idx: u32,
    n_elems: u32,
}

// TODO: try to derive these instead (tricky because of generic T)
impl<T> Clone for BufferHandle<T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}
impl<T> Copy for BufferHandle<T> {}

impl<T> PartialEq for BufferHandle<T> {
    fn eq(&self, o: &Self) -> bool {
        return self.h == o.h
            && self.mutability == o.mutability
            && self.idx == o.idx
            && self.n_elems == o.n_elems;
    }
}
impl<T> Eq for BufferHandle<T> {}

impl<T> std::hash::Hash for BufferHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.h.hash(state);
        self.mutability.hash(state);
        self.idx.hash(state);
        self.n_elems.hash(state);
    }
}

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
}

// Below is backend agnostic (almost...)
pub trait BufferType {
    const USAGE: vk::BufferUsageFlags;
    fn elem_align(&self, _allocator: &AllocatorHandle) -> Option<u16> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct OwningBufferDescriptor<BT> {
    data: Arc<ByteBuffer>,
    mutability: BufferMutability,
    elem_size: u16,
    n_elems: u32,
    buffer_type: BT,
}

impl<BT: BufferType + Clone> BufferDescriptor for OwningBufferDescriptor<BT> {
    type Buffer = TypedBuffer<BT>;
    fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    fn vk_usage_flags(&self) -> vk::BufferUsageFlags {
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
    type Buffer = TypedBuffer<BT>;

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

    fn vk_usage_flags(&self) -> vk::BufferUsageFlags {
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

#[derive(Debug, Clone)]
pub struct UniformBufferType;
impl BufferType for UniformBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
    fn elem_align(&self, allocator: &AllocatorHandle) -> Option<u16> {
        Some(
            allocator
                .get_physical_device_properties()
                .expect("Bad allocator")
                .limits
                .min_uniform_buffer_offset_alignment as u16,
        )
    }
}

pub trait Uniform {}

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

#[derive(Debug, Clone)]
pub struct VertexBufferType {
    format: VertexFormat,
}
impl BufferType for VertexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
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
    pub fn from_raw(data: Vec<u8>, format: VertexFormat, mutability: BufferMutability) -> Self {
        assert_eq!(data.len() as u32 % format.size(), 0);
        let n_elems = data.len() as u32 / format.size();
        let data = unsafe { Arc::new(ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: format.size() as u16,
            mutability,
            buffer_type: VertexBufferType { format },
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone)]
pub struct IndexBufferType {
    index_size: IndexSize,
}

impl BufferType for IndexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::INDEX_BUFFER;
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

#[derive(Debug)]
pub struct TypedBuffer<BT> {
    buffer: DeviceBuffer,
    elem_size: u16,
    stride: u16,
    n_elems: u32,
    buffer_type: BT,
}

impl<BT> TypedBuffer<BT> {
    pub fn create(
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        descriptor: &impl BufferDescriptor,
        buffer_type: BT,
    ) -> Result<(Self, Option<DeviceBuffer>), MemoryError> {
        log::trace!("Creating buffer");
        let elem_size = descriptor.elem_size();
        let elem_align = descriptor.elem_align(allocator);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();
        let n_elems = descriptor.n_elems();

        let (buffer, staging) = match descriptor.mutability() {
            BufferMutability::Immutable => {
                let (buffer, staging) = DeviceBuffer::device_local(
                    allocator,
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
                    allocator,
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
                n_elems,
                buffer_type,
            },
            staging,
        ))
    }

    pub fn recreate(
        &mut self,
        allocator: &AllocatorHandle,
        descriptor: &impl BufferDescriptor,
    ) -> Result<(), MemoryError> {
        assert!(descriptor.mutability() == BufferMutability::Mutable);
        let elem_size = descriptor.elem_size();
        let n_elems = descriptor.n_elems();
        let elem_align = descriptor.elem_align(allocator);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();
        let stride = stride(elem_size, elem_align);
        let size = stride as usize * n_elems as usize;
        if size > self.buffer.size() {
            self.buffer = DeviceBuffer::persistent_mapped(
                allocator,
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

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }

    pub fn buffer_type(&self) -> &BT {
        &self.buffer_type
    }

    pub fn stride(&self) -> u16 {
        self.stride
    }
}

pub type VertexBuffer = TypedBuffer<VertexBufferType>;
impl VertexBuffer {
    pub fn format(&self) -> &VertexFormat {
        &self.buffer_type().format
    }
}

pub type IndexBuffer = TypedBuffer<IndexBufferType>;
impl IndexBuffer {
    pub fn vk_index_type(&self) -> vk::IndexType {
        vk::IndexType::from(self.buffer_type().index_size)
    }
}

pub type UniformBuffer = TypedBuffer<UniformBufferType>;
impl UniformBuffer {
    pub fn update_with<T: Uniform + Copy>(
        &mut self,
        data: &T,
        idx: u64,
    ) -> Result<(), MemoryError> {
        let raw_data = as_bytes(data);
        let offset = (idx * self.stride() as u64) as usize;
        self.buffer_mut().update_data_at(raw_data, offset)
    }

    pub fn size(&self) -> u64 {
        assert!(self.elem_size() <= self.stride());
        self.stride() as u64 * self.n_elems() as u64
    }
}

struct HostBuffer<BT> {
    buffer: Arc<ByteBuffer>,
    buffer_type: BT,
}
