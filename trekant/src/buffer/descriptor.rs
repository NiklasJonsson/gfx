use super::{
    BufferMutability, BufferType, DeviceBuffer, HostBuffer, IndexBufferType, IndexInt,
    UniformBufferType, VertexBufferType,
};
use crate::backend;

use crate::traits::Uniform;
use crate::util::{as_byte_slice, ByteBuffer};
use backend::buffer::Buffer;
use backend::command::CommandBuffer;
use backend::{AllocatorHandle, MemoryError};

use crate::vk;
use crate::vertex::VertexDefinition;

use std::sync::Arc;

pub struct BufferResult<B> {
    pub buffer: B,
    pub transient: Option<Buffer>,
}

#[derive(Debug)]
enum DescriptorData<'a> {
    Owned(ByteBuffer),
    Shared(Arc<ByteBuffer>),
    Borrowed(&'a [u8]),
}

pub struct BufferDescriptor<'a, BT> {
    data: DescriptorData<'a>,
    mutability: BufferMutability,
    n_elems: u32,
    buffer_type: BT,
}

impl<BT: Clone + BufferType> BufferDescriptor<'static, BT> {
    pub fn from_host_buffer(hb: &HostBuffer<BT>, mutability: BufferMutability) -> Self {
        Self {
            data: DescriptorData::Shared(hb.data.clone()),
            mutability,
            n_elems: hb.n_elems,
            buffer_type: hb.buffer_type.clone(),
        }
    }
}

pub type UniformBufferDescriptor<'a> = BufferDescriptor<'a, UniformBufferType>;
pub type VertexBufferDescriptor<'a> = BufferDescriptor<'a, VertexBufferType>;
pub type IndexBufferDescriptor<'a> = BufferDescriptor<'a, IndexBufferType>;

macro_rules! impl_descriptor_from {
    ($name:ident, $trait:ident, $buffer_type:ident) => {
        impl $name<'static> {
            pub fn from_vec<T: Copy + $trait + 'static>(
                data: Vec<T>,
                mutability: BufferMutability,
            ) -> Self {
                let n_elems = data.len() as u32;
                let data = DescriptorData::Owned(unsafe { ByteBuffer::from_vec(data) });
                let buffer_type = $buffer_type::from_trait::<T>();
                Self {
                    data,
                    n_elems,
                    mutability,
                    buffer_type,
                }
            }

            pub fn from_single<T: Copy + $trait + 'static>(
                t: T,
                mutability: BufferMutability,
            ) -> Self {
                Self::from_vec(vec![t], mutability)
            }

            /// # Safety
            /// The contents of the vector must match the description of it in buffer_type. Furthermore, the contents must fulfill vulkan requirements, e.g. alignment,
            /// for the supplied buffer type.
            pub unsafe fn from_raw(
                data: Vec<u8>,
                buffer_type: $buffer_type,
                mutability: BufferMutability,
            ) -> Self {
                assert_eq!(data.len() % buffer_type.elem_size() as usize, 0);
                let n_elems = data.len() as u32 / buffer_type.elem_size() as u32;
                let data = DescriptorData::Owned(ByteBuffer::from_vec(data));
                Self {
                    data,
                    n_elems,
                    mutability,
                    buffer_type,
                }
            }
        }

        impl<'a> $name<'a> {
            pub fn from_slice<T: Copy + $trait + 'static>(
                data: &'a [T],
                mutability: BufferMutability,
            ) -> Self {
                let n_elems = data.len() as u32;
                let data = DescriptorData::Borrowed(unsafe { as_byte_slice(data) });
                let buffer_type = $buffer_type::from_trait::<T>();
                Self {
                    data,
                    n_elems,
                    mutability,
                    buffer_type,
                }
            }
        }
    };
}

impl_descriptor_from!(UniformBufferDescriptor, Uniform, UniformBufferType);
impl_descriptor_from!(VertexBufferDescriptor, VertexDefinition, VertexBufferType);
impl_descriptor_from!(IndexBufferDescriptor, IndexInt, IndexBufferType);

impl<'a, BT> BufferDescriptor<'a, BT> {
    pub fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }

    pub fn data(&self) -> &[u8] {
        match &self.data {
            DescriptorData::Borrowed(slice) => slice,
            DescriptorData::Owned(buf) => buf,
            DescriptorData::Shared(buf) => buf,
        }
    }
}

impl<'a, BT: BufferType> BufferDescriptor<'a, BT> {
    pub fn vk_usage_flags(&self) -> vk::BufferUsageFlags {
        BT::USAGE
    }

    pub fn elem_size(&self) -> u16 {
        self.buffer_type.elem_size()
    }

    pub fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        self.buffer_type
            .elem_align(allocator)
            .unwrap_or_else(|| self.elem_size())
    }
}

impl<'a, BT: BufferType + Clone> BufferDescriptor<'a, BT> {
    pub fn enqueue_single(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<DeviceBuffer<BT>>, MemoryError> {
        let (buffer, transient) =
            DeviceBuffer::<BT>::create(allocator, command_buffer, self, self.buffer_type.clone())?;

        Ok(BufferResult { buffer, transient })
    }

    #[allow(clippy::type_complexity)]
    pub fn enqueue(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<
        (
            BufferResult<DeviceBuffer<BT>>,
            Option<BufferResult<DeviceBuffer<BT>>>,
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
