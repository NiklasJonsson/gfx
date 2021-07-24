mod buffer_storage;
mod descriptor;

use crate::backend;
use crate::resource::Handle;
use crate::vertex::VertexFormat;
use backend::command::CommandBuffer;
use backend::{buffer::Buffer, util::stride, AllocatorHandle, MemoryError};

use crate::util::{as_bytes, ByteBuffer};
use crate::vertex::VertexDefinition;

use ash::vk;

use std::sync::Arc;

pub use descriptor::BufferDescriptor;

// TODO: Remove
pub use descriptor::{
    OwningBufferDescriptor, OwningIndexBufferDescriptor, OwningUniformBufferDescriptor,
    OwningVertexBufferDescriptor,
};

pub use buffer_storage::DrainIterator;
use buffer_storage::{AsyncDeviceBufferStorage, DeviceBufferStorage};
pub type UniformBuffers = DeviceBufferStorage<DeviceUniformBuffer>;
pub type AsyncUniformBuffers = AsyncDeviceBufferStorage<DeviceUniformBuffer>;
pub type VertexBuffers = DeviceBufferStorage<DeviceVertexBuffer>;
pub type AsyncVertexBuffers = AsyncDeviceBufferStorage<DeviceVertexBuffer>;
pub type IndexBuffers = DeviceBufferStorage<DeviceIndexBuffer>;
pub type AsyncIndexBuffers = AsyncDeviceBufferStorage<DeviceIndexBuffer>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferMutability {
    Immutable,
    Mutable,
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

pub trait BufferType {
    const USAGE: vk::BufferUsageFlags;
    fn elem_align(&self, _allocator: &AllocatorHandle) -> Option<u16> {
        None
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

// TODO: This should have a corresponding derive that exposes align/size/padding checks
/// Marker trait for a type that is a uniform
pub trait Uniform {}

#[derive(Debug, Clone)]
pub struct VertexBufferType {
    format: VertexFormat,
}
impl BufferType for VertexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
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

#[derive(Debug)]
pub struct DeviceBuffer<BT> {
    buffer: Buffer,
    elem_size: u16,
    n_elems: u32,
    buffer_type: BT,
    stride: u16,
}

impl<BT> DeviceBuffer<BT> {
    pub fn create(
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        descriptor: &impl BufferDescriptor,
        buffer_type: BT,
    ) -> Result<(Self, Option<Buffer>), MemoryError> {
        log::trace!("Creating buffer");
        let elem_size = descriptor.elem_size();
        let elem_align = descriptor.elem_align(allocator);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();
        let n_elems = descriptor.n_elems();

        let (buffer, staging) = match descriptor.mutability() {
            BufferMutability::Immutable => {
                let (buffer, staging) = Buffer::device_local(
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
                Buffer::persistent_mapped(
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
            self.buffer = Buffer::persistent_mapped(
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

    pub fn buffer_mut(&mut self) -> &mut Buffer {
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

pub type DeviceVertexBuffer = DeviceBuffer<VertexBufferType>;
impl DeviceVertexBuffer {
    pub fn format(&self) -> &VertexFormat {
        &self.buffer_type().format
    }
}

pub type DeviceIndexBuffer = DeviceBuffer<IndexBufferType>;
impl DeviceIndexBuffer {
    pub fn vk_index_type(&self) -> vk::IndexType {
        vk::IndexType::from(self.buffer_type().index_size)
    }
}

pub type DeviceUniformBuffer = DeviceBuffer<UniformBufferType>;
impl DeviceUniformBuffer {
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

#[derive(Debug, Clone)]
pub struct HostBuffer<BT> {
    data: Arc<ByteBuffer>,
    elem_size: u16,
    n_elems: u32,
    buffer_type: BT,
}

pub type HostVertexBuffer = HostBuffer<VertexBufferType>;
pub type HostIndexBuffer = HostBuffer<IndexBufferType>;
pub type HostUniformBuffer = HostBuffer<UniformBufferType>;

impl HostUniformBuffer {
    pub fn from_vec<T: Copy + Uniform + 'static>(data: Vec<T>) -> Self {
        let n_elems = data.len() as u32;
        let data = unsafe { Arc::new(ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: std::mem::size_of::<T>() as u16,
            buffer_type: UniformBufferType,
        }
    }
}

impl HostVertexBuffer {
    pub fn from_vec<V: VertexDefinition + Copy + 'static>(data: Vec<V>) -> Self {
        let n_elems = data.len() as u32;
        let data = unsafe { Arc::new(ByteBuffer::from_vec(data)) };
        Self {
            data,
            n_elems,
            elem_size: std::mem::size_of::<V>() as u16,
            buffer_type: VertexBufferType {
                format: V::format(),
            },
        }
    }

    pub unsafe fn from_raw(data: Vec<u8>, format: VertexFormat) -> Self {
        assert_eq!(data.len() as u32 % format.size(), 0);
        let n_elems = data.len() as u32 / format.size();
        let data = Arc::new(ByteBuffer::from_vec(data));
        Self {
            data,
            n_elems,
            elem_size: format.size() as u16,
            buffer_type: VertexBufferType { format },
        }
    }
}

pub trait IndexInt {
    fn size() -> IndexSize;
}

impl IndexInt for u16 {
    fn size() -> IndexSize {
        IndexSize::Size16
    }
}

impl IndexInt for u32 {
    fn size() -> IndexSize {
        IndexSize::Size32
    }
}

impl HostIndexBuffer {
    pub fn from_vec<T: IndexInt + Copy + 'static>(data: Vec<T>) -> Self {
        // TODO: static assert here
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
            buffer_type: IndexBufferType { index_size },
        }
    }
}
