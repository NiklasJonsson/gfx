mod descriptor;

use crate::resource::{BufferedStorage, Handle, Storage};

use crate::vertex::VertexFormat;
use crate::{backend, util, Std140};
use backend::command::CommandBuffer;
use backend::{buffer::Buffer, util::compute_stride, AllocatorHandle, MemoryError};

use crate::traits::Uniform;
use crate::vertex::VertexDefinition;

use ash::vk;

use std::convert::TryInto;

pub use descriptor::BufferDescriptor;
pub use descriptor::BufferLayout;

pub type Buffers = DeviceBufferStorage;
pub type AsyncBuffers = AsyncDeviceBufferStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferTypeId(pub std::mem::Discriminant<BufferType>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferMutability {
    Immutable,
    Mutable,
}

// TODO: Revisit this! It's only used in ram?
pub trait BufferTypeTrait {
    const USAGE: vk::BufferUsageFlags;
    fn elem_align(&self) -> u16;
    fn elem_size(&self) -> u16;
    fn buffer_type(&self) -> BufferType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    h: Handle<DeviceBuffer>,
    mutability: BufferMutability,
    offset: u32,
    len: u32,
    ty: BufferTypeId,
}

impl BufferHandle {
    pub fn sub_buffer(h: Self, idx: u32, n_elems: u32) -> Self {
        assert!((idx + n_elems) <= (h.offset + h.len));
        Self {
            offset: idx,
            len: n_elems,
            ..h
        }
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
            offset: idx,
            len: n_elems,
            ty,
        }
    }

    pub fn take_first(&mut self, n: u32) -> BufferHandle {
        assert!(self.len >= n);
        let idx = self.offset;

        self.offset += n;
        self.len -= n;

        BufferHandle {
            offset: idx,
            len: n,
            ..*self
        }
    }

    pub fn slice(&self, start: u32, len: u32) -> BufferHandle {
        BufferHandle {
            offset: self.offset + start,
            len,
            ..*self
        }
    }

    pub fn handle(&self) -> &Handle<DeviceBuffer> {
        &self.h
    }

    pub fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn end(&self) -> u32 {
        self.offset + self.len
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn buffer_type_id(&self) -> BufferTypeId {
        self.ty
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncBufferHandle {
    h: Handle<resurs::Async<DeviceBuffer>>,
    mutability: BufferMutability,
    idx: u32,
    n_elems: u32,
    ty: BufferTypeId,
}

impl AsyncBufferHandle {
    pub fn sub_buffer(h: Self, idx: u32, n_elems: u32) -> Self {
        assert!((idx + n_elems) <= (h.idx + h.n_elems));
        Self { idx, n_elems, ..h }
    }

    /// # Safety
    /// The handle must refer to a valid buffer resource. idx + n_elems must be less than or equal to the number of elements the buffer that the handle refers to was created with.
    pub(crate) unsafe fn from_buffer(
        handle: Handle<resurs::Async<DeviceBuffer>>,
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

    pub fn base_buffer(&self) -> Handle<resurs::Async<DeviceBuffer>> {
        self.h
    }

    pub fn idx(&self) -> u32 {
        self.idx
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UniformBufferType {
    elem_size: u16,
    elem_align: u16,
}

impl BufferTypeTrait for UniformBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::UNIFORM_BUFFER;
    fn elem_align(&self) -> u16 {
        self.elem_align
    }
    fn elem_size(&self) -> u16 {
        self.elem_size
    }

    fn buffer_type(&self) -> BufferType {
        BufferType::Uniform(*self)
    }
}

impl UniformBufferType {
    pub fn as_enum<U: Uniform>() -> BufferType {
        BufferType::Uniform(Self::from_type::<U>())
    }

    pub fn from_type<U: Uniform>() -> Self {
        Self {
            elem_size: U::size(),
            elem_align: U::align(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VertexBufferType {
    format: VertexFormat,
}

impl VertexBufferType {
    pub fn format(&self) -> &VertexFormat {
        &self.format
    }
}

impl BufferTypeTrait for VertexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::VERTEX_BUFFER;
    fn elem_align(&self) -> u16 {
        self.elem_size()
    }
    fn elem_size(&self) -> u16 {
        self.format
            .size()
            .try_into()
            .expect("Vertex format is too big")
    }

    fn buffer_type(&self) -> BufferType {
        BufferType::Vertex(self.clone())
    }
}

impl VertexBufferType {
    pub fn as_enum<V: VertexDefinition>() -> BufferType {
        BufferType::Vertex(Self::from_type::<V>())
    }

    pub fn from_type<V: VertexDefinition>() -> Self {
        Self::new(V::format())
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

#[derive(Debug, Clone, Copy)]
pub struct IndexBufferType {
    size: IndexSize,
}

impl BufferTypeTrait for IndexBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::INDEX_BUFFER;
    fn elem_align(&self) -> u16 {
        self.elem_size()
    }
    fn elem_size(&self) -> u16 {
        self.size.size() as u16
    }

    fn buffer_type(&self) -> BufferType {
        BufferType::Index(*self)
    }
}

impl IndexBufferType {
    pub fn as_enum<I: IndexInt>() -> BufferType {
        BufferType::Index(Self::from_type::<I>())
    }

    pub fn from_type<I: IndexInt>() -> Self {
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
    elem_size: u16,
    elem_align: u16,
}

impl BufferTypeTrait for StorageBufferType {
    const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::STORAGE_BUFFER;
    fn elem_align(&self) -> u16 {
        self.elem_align
    }
    fn elem_size(&self) -> u16 {
        self.elem_size
    }

    fn buffer_type(&self) -> BufferType {
        BufferType::Storage(*self)
    }
}

impl StorageBufferType {
    pub fn as_enum<T: Std140>() -> BufferType {
        BufferType::Storage(Self::from_type::<T>())
    }

    pub fn from_type<T: Std140>() -> Self {
        Self {
            elem_size: <T as Std140>::SIZE
                .try_into()
                .expect("Failed to narrow size"),
            elem_align: <T as Std140>::ALIGNMENT
                .try_into()
                .expect("Alignment is too big, doesn't fit in u16"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum BufferType {
    Vertex(VertexBufferType),
    Index(IndexBufferType),
    Uniform(UniformBufferType),
    Storage(StorageBufferType),
}

impl BufferType {
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

    pub fn elem_align(&self) -> u16 {
        match self {
            Self::Index(ty) => ty.elem_align(),
            Self::Vertex(ty) => ty.elem_align(),
            Self::Uniform(ty) => ty.elem_align(),
            Self::Storage(s) => s.elem_align(),
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
    buffer_type: BufferType,
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
        let elem_align = descriptor.elem_align();
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
        let elem_align = descriptor.elem_align();
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

        if (offset + data.len()) > self.buffer.size() {
            panic!(
                "Data of {} bytes does not fit into buffer ({:p}) at {} sized {}",
                data.len(),
                self.buffer.ptr(),
                offset,
                self.buffer.size(),
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

    pub fn buffer_type(&self) -> &BufferType {
        &self.buffer_type
    }

    pub fn stride(&self) -> u16 {
        compute_stride(self.elem_size, self.elem_align)
    }
}

impl DeviceBuffer {
    pub fn vk_index_type(&self) -> Option<vk::IndexType> {
        match &self.buffer_type {
            BufferType::Index(idx) => Some(vk::IndexType::from(idx.size)),
            _ => None,
        }
    }
}
#[derive(Default)]
pub struct DeviceBufferStorage {
    buffered: BufferedStorage<DeviceBuffer>,
    unbuffered: Storage<DeviceBuffer>,
}

impl DeviceBufferStorage {
    pub fn get_all(&self, h: BufferHandle) -> Option<(&DeviceBuffer, Option<&DeviceBuffer>)> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()).map(|x| (x, None)),
            BufferMutability::Mutable => {
                self.buffered.get_all(h.handle()).map(|[x, y]| (x, Some(y)))
            }
        }
    }

    pub fn get_all_mut(
        &mut self,
        h: BufferHandle,
    ) -> Option<(&mut DeviceBuffer, Option<&mut DeviceBuffer>)> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get_mut(h.handle()).map(|x| (x, None)),
            BufferMutability::Mutable => self
                .buffered
                .get_all_mut(h.handle())
                .map(|[x, y]| (x, Some(y))),
        }
    }

    pub fn get_buffered(&self, h: BufferHandle, idx: usize) -> Option<&DeviceBuffer> {
        assert_eq!(h.mutability(), BufferMutability::Mutable);
        self.buffered.get(h.handle(), idx)
    }

    pub fn get_buffered_mut(&mut self, h: BufferHandle, idx: usize) -> Option<&mut DeviceBuffer> {
        assert_eq!(h.mutability(), BufferMutability::Mutable);
        self.buffered.get_mut(h.handle(), idx)
    }

    pub fn get_unbuffered(&self, h: BufferHandle) -> Option<&DeviceBuffer> {
        assert_eq!(h.mutability(), BufferMutability::Immutable);
        self.unbuffered.get(h.handle())
    }

    // TODO: BufferHandle here. Needs buffer type which exposes sizes etc.
    pub fn add(
        &mut self,
        data0: DeviceBuffer,
        data1: Option<DeviceBuffer>,
    ) -> resurs::Handle<DeviceBuffer> {
        match data1 {
            Some(data1) => self.buffered.add([data0, data1]),
            None => self.unbuffered.add(data0),
        }
    }

    pub fn get(&self, h: BufferHandle, idx: usize) -> Option<&DeviceBuffer> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()),
            BufferMutability::Mutable => self.buffered.get(h.handle(), idx),
        }
    }

    pub fn get_mut(&mut self, h: BufferHandle, idx: usize) -> Option<&mut DeviceBuffer> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get_mut(h.handle()),
            BufferMutability::Mutable => self.buffered.get_mut(h.handle(), idx),
        }
    }

    pub fn has(&self, h: BufferHandle) -> bool {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.has(h.handle()),
            BufferMutability::Mutable => self.buffered.has(h.handle()),
        }
    }

    pub fn drain_filter<F1, F2>(&mut self, f1: F1, f2: F2) -> DrainFilter<'_, F1, F2, DeviceBuffer>
    where
        F1: FnMut(&mut DeviceBuffer) -> bool,
        F2: FnMut(&mut [DeviceBuffer; 2]) -> bool,
    {
        DrainFilter {
            unbuffered_iter: self.unbuffered.drain_filter(f1),
            buffered_iter: self.buffered.drain_filter(f2),
        }
    }
}

pub struct DrainFilter<'a, F1, F2, DeviceBuffer>
where
    F1: FnMut(&mut DeviceBuffer) -> bool,
    F2: FnMut(&mut [DeviceBuffer; 2]) -> bool,
{
    unbuffered_iter: resurs::storage::DrainFilter<'a, F1, DeviceBuffer>,
    buffered_iter: resurs::storage::DrainFilter<'a, F2, [DeviceBuffer; 2]>,
}

impl<'a, F1, F2, DeviceBuffer> Iterator for DrainFilter<'a, F1, F2, DeviceBuffer>
where
    F1: FnMut(&mut DeviceBuffer) -> bool,
    F2: FnMut(&mut [DeviceBuffer; 2]) -> bool,
{
    type Item = (Handle<DeviceBuffer>, DeviceBuffer, Option<DeviceBuffer>);
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if let Some((handle, item)) = self.unbuffered_iter.next() {
            return Some((handle, item, None));
        }

        if let Some((handle, [item0, item1])) = self.buffered_iter.next() {
            return Some((handle.as_unbuffered(), item0, Some(item1)));
        }

        None
    }
}

use crate::resource::Async;

#[derive(Default)]
pub struct AsyncDeviceBufferStorage {
    buffered: BufferedStorage<Async<DeviceBuffer>>,
    unbuffered: Storage<Async<DeviceBuffer>>,
}

impl AsyncDeviceBufferStorage {
    pub fn allocate(&mut self, desc: &BufferDescriptor<'_>) -> AsyncBufferHandle {
        let raw_handle = match desc.mutability() {
            BufferMutability::Immutable => self.unbuffered.add(Async::<DeviceBuffer>::Pending),
            BufferMutability::Mutable => self.buffered.add([
                Async::<DeviceBuffer>::Pending,
                Async::<DeviceBuffer>::Pending,
            ]),
        };
        unsafe {
            AsyncBufferHandle::from_buffer(
                raw_handle,
                0,
                desc.n_elems(),
                desc.mutability(),
                desc.buffer_type().ty(),
            )
        }
    }

    pub fn insert(&mut self, h: AsyncBufferHandle, buf0: DeviceBuffer, buf1: Option<DeviceBuffer>) {
        match h.mutability {
            BufferMutability::Immutable => {
                let loc: &mut Async<DeviceBuffer> = self
                    .unbuffered
                    .get_mut(&h.h)
                    .expect("Expected handle to be allocated");
                *loc = Async::Available(buf0);
            }
            BufferMutability::Mutable => {
                let loc = self
                    .buffered
                    .get_all_mut(&h.h)
                    .expect("Expected handle to be allocated");
                loc[0] = Async::Available(buf0);
                loc[1] = Async::Available(buf1.expect("Mutable buffers require two buffers"));
            }
        }
    }

    pub fn drain_available(&mut self) -> DrainIterator<'_, DeviceBuffer> {
        let f1 = |x: &mut Async<DeviceBuffer>| std::matches!(x, Async::Available(_));
        let f2 = |x: &mut [Async<DeviceBuffer>; 2]| std::matches!(x[0], Async::Available(_));
        DrainFilter {
            unbuffered_iter: self.unbuffered.drain_filter(f1),
            buffered_iter: self.buffered.drain_filter(f2),
        }
    }
}

pub type DrainIterator<'a, T> =
    DrainFilter<'a, fn(&mut Async<T>) -> bool, fn(&mut [Async<T>; 2]) -> bool, Async<T>>;
