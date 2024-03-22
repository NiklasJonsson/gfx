mod buffer_storage;
mod descriptor;

use crate::resource::Handle;
use crate::vertex::VertexFormat;
use crate::{backend, util, Std140};
use backend::command::CommandBuffer;
use backend::{buffer::Buffer, util::compute_stride, AllocatorHandle, MemoryError};

use crate::traits::Uniform;
use crate::util::ByteBuffer;
use crate::vertex::VertexDefinition;

use ash::vk;

use std::convert::TryInto;
use std::sync::Arc;

pub use descriptor::BufferDescriptor;

pub use buffer_storage::DrainIterator;
use buffer_storage::{AsyncDeviceBufferStorage, DeviceBufferStorage};
pub type Buffers = DeviceBufferStorage;
pub type AsyncBuffers = AsyncDeviceBufferStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferTypeId(pub std::mem::Discriminant<BufferTypeDesc>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferMutability {
    Immutable,
    Mutable,
}

pub trait BufferType {
    const USAGE: vk::BufferUsageFlags;
    fn elem_align(&self, _allocator: &AllocatorHandle) -> u16 {
        self.elem_size()
    }
    fn elem_size(&self) -> u16;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    h: Handle<DeviceBuffer>,
    mutability: BufferMutability,
    idx: u32,
    n_elems: u32,
    ty: BufferTypeId,
}

impl BufferHandle {
    pub fn sub_buffer(h: Self, idx: u32, n_elems: u32) -> Self {
        assert!((idx + n_elems) <= (h.idx + h.n_elems));
        Self { idx, n_elems, ..h }
    }

    /// # Safety
    /// The handle must refer to a valid buffer resource. idx + n_elems must be less than or equal to the number of elements the buffer that the handle refers to was created with.
    pub unsafe fn from_buffer(
        handle: Handle<DeviceBuffer>,
        idx: u32,
        n_elems: u32,
        mutability: BufferMutability,
        ty: BufferTypeId,
    ) -> Self {
        Self {
            h: handle,
            mutability,
            idx,
            n_elems,
            ty,
        }
    }

    pub fn handle(&self) -> &Handle<DeviceBuffer> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncBufferHandle {
    h: Handle<resurs::Async<DeviceBuffer>>,
    mutability: BufferMutability,
    n_elems: u32,
    ty: BufferTypeId,
}

impl AsyncBufferHandle {
    /// # Safety
    /// The handle must refer to a valid buffer resource. idx + n_elems must be less than or equal to the number of elements the buffer that the handle refers to was created with.
    pub unsafe fn from_buffer(
        handle: Handle<resurs::Async<DeviceBuffer>>,
        mutability: BufferMutability,
        n_elems: u32,
        ty: BufferTypeId,
    ) -> Self {
        Self {
            h: handle,
            mutability,
            n_elems,
            ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UniformBufferType {
    elem_size: u16,
}
impl BufferType for UniformBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
    fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        const REQUIRED_ALIGNMENT_SUPPORT: u16 = 256;
        let props = unsafe { allocator.get_physical_device_properties() }.expect("Bad allocator");
        let alignment = props.limits.min_uniform_buffer_offset_alignment;
        if alignment == 0 {
            REQUIRED_ALIGNMENT_SUPPORT
        } else {
            alignment
                .try_into()
                .expect("Alignment requirement is too big")
        }
    }
    fn elem_size(&self) -> u16 {
        self.elem_size
    }
}

impl UniformBufferType {
    fn from_trait<U: Uniform>() -> Self {
        Self {
            elem_size: U::size(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VertexBufferType {
    format: VertexFormat,
}
impl BufferType for VertexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
    fn elem_size(&self) -> u16 {
        self.format
            .size()
            .try_into()
            .expect("Vertex format is too big")
    }
}

impl VertexBufferType {
    fn from_trait<V: VertexDefinition>() -> Self {
        Self {
            format: V::format(),
        }
    }

    pub fn new(format: VertexFormat) -> Self {
        Self { format }
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
    size: IndexSize,
}

impl BufferType for IndexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::INDEX_BUFFER;
    fn elem_size(&self) -> u16 {
        self.size.size() as u16
    }
}

impl IndexBufferType {
    fn from_trait<I: IndexInt>() -> Self {
        Self { size: I::size() }
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

#[derive(Debug, Clone, Copy)]
pub struct StorageBufferType {
    size: u16,
}

impl BufferType for StorageBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
    fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        const REQUIRED_ALIGNMENT_SUPPORT: u16 = 256;
        let props = unsafe { allocator.get_physical_device_properties() }.expect("Bad allocator");
        let alignment = props.limits.min_storage_buffer_offset_alignment;
        if alignment == 0 {
            REQUIRED_ALIGNMENT_SUPPORT
        } else {
            alignment
                .try_into()
                .expect("Alignment requirement is too big")
        }
    }
    fn elem_size(&self) -> u16 {
        self.size
    }
}

impl StorageBufferType {
    fn from_trait<T: Std140>() -> Self {
        Self {
            size: <T as Std140>::SIZE
                .try_into()
                .expect("Failed to narrow size"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BufferTypeDesc {
    Vertex(VertexBufferType),
    Index(IndexBufferType),
    Uniform(UniformBufferType),
    Storage(StorageBufferType),
}

impl BufferTypeDesc {
    pub fn ty(&self) -> BufferTypeId {
        BufferTypeId(std::mem::discriminant(self))
    }

    pub fn elem_size(&self) -> u16 {
        match self {
            Self::Index(idx) => idx.elem_size(),
            Self::Vertex(vertex) => vertex.elem_size(),
            Self::Uniform(uni) => uni.elem_size(),
            Self::Storage(s) => s.elem_size(),
        }
    }

    pub fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        match self {
            Self::Index(ty) => ty.elem_align(allocator),
            Self::Vertex(ty) => ty.elem_align(allocator),
            Self::Uniform(ty) => ty.elem_align(allocator),
            Self::Storage(s) => s.elem_align(allocator),
        }
    }
}

#[derive(Debug)]
pub struct DeviceBuffer {
    buffer: Buffer,
    n_elems: u32,
    elem_size: u16,
    elem_align: u16,
    mutability: BufferMutability,
    buffer_type: BufferTypeDesc,
}

impl DeviceBuffer {
    /// NOTE: The returned staging buffer needs to outlive the command buffer!
    /// TODO: This should probably be unsafe or handled by the API...
    pub fn create(
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
        descriptor: &BufferDescriptor,
    ) -> Result<(Self, Option<Buffer>), MemoryError> {
        log::trace!("Creating buffer");
        let elem_size = descriptor.elem_size();
        let elem_align = descriptor.elem_align(allocator);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();
        let n_elems = descriptor.n_elems();
        let buffer_type = descriptor.buffer_type().clone();

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

        Ok((
            Self {
                buffer,
                n_elems,
                buffer_type,
                elem_align,
                elem_size,
                mutability: descriptor.mutability(),
            },
            staging,
        ))
    }

    pub fn recreate(
        &mut self,
        allocator: &AllocatorHandle,
        descriptor: &BufferDescriptor,
    ) -> Result<(), MemoryError> {
        assert!(descriptor.mutability() == BufferMutability::Mutable);
        assert!(self.mutability == BufferMutability::Mutable);
        assert_eq!(self.buffer_type().ty(), descriptor.buffer_type().ty());
        let elem_size = descriptor.elem_size();
        let n_elems = descriptor.n_elems();
        let elem_align = descriptor.elem_align(allocator);
        let vk_buffer_usage_flags = descriptor.vk_usage_flags();
        let data = descriptor.data();
        let stride = compute_stride(elem_size, elem_align);
        let size = stride as usize * n_elems as usize;

        // NOTE: The below seems a bit incorrect, what about the capacity that is lost when we overwrite the n_elems?
        // It seems like it would lead to unaccessiable memory if we keep succesively using less and less of the buffer?
        // TODO: Revisit this! We might want some kind of a resize/reserve/realloc API instead.
        if size > self.buffer.size() {
            self.buffer = Buffer::persistent_mapped(
                allocator,
                vk_buffer_usage_flags,
                data,
                elem_size,
                stride,
            )?
        } else {
            self.write(0, descriptor.data());
        };

        self.n_elems = n_elems;
        self.elem_align = elem_align;
        self.elem_size = elem_size;
        Ok(())
    }

    /// Write the 'data' at 'offset' in this buffer.
    /// # Panic:
    /// * If the buffer is not mutable.
    /// * If the data does not fit.
    pub fn write(&mut self, offset: usize, data: &[u8]) {
        if self.mutability != BufferMutability::Mutable {
            panic!(
                "Buffer at {:p} is not mutable, can't write to it",
                self.buffer.ptr()
            );
        }

        if (offset + data.len()) >= self.buffer.size() {
            panic!(
                "Data of {} bytes does not fit in buffer {:p}, at {offset}",
                data.len(),
                self.buffer.ptr()
            );
        }

        let dst_start = self.buffer.mut_ptr();
        unsafe {
            let dst = dst_start.add(offset);
            util::copy_nonoverlapping_aligned(data, dst, self.elem_size, self.elem_align);
        }
    }

    pub fn buffer_mut(&mut self) -> &mut Buffer {
        &mut self.buffer
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.vk_buffer()
    }

    pub fn elem_size(&self) -> u16 {
        self.buffer_type.elem_size()
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }

    pub fn buffer_type(&self) -> &BufferTypeDesc {
        &self.buffer_type
    }

    pub fn stride(&self) -> u16 {
        compute_stride(self.elem_size, self.elem_align)
    }
}

impl DeviceBuffer {
    pub fn vk_index_type(&self) -> Option<vk::IndexType> {
        match &self.buffer_type {
            BufferTypeDesc::Index(idx) => Some(vk::IndexType::from(idx.size)),
            _ => None,
        }
    }
}

/* pub type DeviceVertexBuffer = DeviceBuffer<VertexBufferType>;
impl DeviceVertexBuffer {
    pub fn vertex_format(&self) -> &VertexFormat {
        &self.buffer_type().format
    }
}

pub type DeviceIndexBuffer = DeviceBuffer<IndexBufferType>;
pub type DeviceUniformBuffer = DeviceBuffer<UniformBufferType>;
impl DeviceUniformBuffer {
    pub fn update_with<T: Uniform>(&mut self, data: &T, idx: u64) {

    }

    pub fn size(&self) -> u64 {
        assert!(self.elem_size() <= self.stride());
        self.stride() as u64 * self.n_elems() as u64
    }
}

pub type DeviceStorageBuffer = DeviceBuffer<StorageBufferType>;
*/

#[derive(Debug, Clone)]
pub struct HostBuffer<BT> {
    data: Arc<ByteBuffer>,
    n_elems: u32,
    buffer_type: BT,
}

pub type HostVertexBuffer = HostBuffer<VertexBufferType>;
pub type HostIndexBuffer = HostBuffer<IndexBufferType>;
pub type HostUniformBuffer = HostBuffer<UniformBufferType>;
pub type HostStorageBuffer = HostBuffer<StorageBufferType>;

macro_rules! impl_host_buffer_from {
    ($name:ty, $trait:ident, $buffer_type:ident) => {
        impl $name {
            pub fn from_vec<T: Copy + $trait + 'static>(data: Vec<T>) -> Self {
                let n_elems = data.len() as u32;
                let data = Arc::new(unsafe { ByteBuffer::from_vec(data) });
                let buffer_type = $buffer_type::from_trait::<T>();
                Self {
                    data,
                    n_elems,
                    buffer_type,
                }
            }

            pub fn from_single<T: Copy + $trait + 'static>(t: T) -> Self {
                Self::from_vec(vec![t])
            }

            /// # Safety
            /// The contents of the vector must match the description of it in buffer_type. Furthermore, the contents must fulfill vulkan requirements, e.g. alignment,
            /// for the supplied buffer type.
            pub unsafe fn from_raw(data: Vec<u8>, buffer_type: $buffer_type) -> Self {
                assert_eq!(data.len() % buffer_type.elem_size() as usize, 0);
                let n_elems = data.len() as u32 / buffer_type.elem_size() as u32;
                let data = Arc::new(ByteBuffer::from_vec(data));
                Self {
                    data,
                    n_elems,
                    buffer_type,
                }
            }
        }
    };
}

impl_host_buffer_from!(HostUniformBuffer, Uniform, UniformBufferType);
impl_host_buffer_from!(HostVertexBuffer, VertexDefinition, VertexBufferType);
impl_host_buffer_from!(HostIndexBuffer, IndexInt, IndexBufferType);

impl HostVertexBuffer {
    pub fn format(&self) -> &VertexFormat {
        &self.buffer_type.format
    }
}
