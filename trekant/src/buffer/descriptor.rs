use super::{
    BufferMutability, BufferType, BufferTypeDesc, DeviceBuffer, HostBuffer, IndexBufferType,
    IndexInt, StorageBufferType, UniformBufferType, VertexBufferType,
};
use crate::{backend, Std140};

use crate::traits::Uniform;
use crate::util::{as_byte_slice, ByteBuffer};
use backend::buffer::Buffer;
use backend::command::CommandBuffer;
use backend::{AllocatorHandle, MemoryError};

use crate::descriptor::DescriptorData;
use crate::vertex::VertexDefinition;
use crate::vk;

pub struct BufferResult<B> {
    pub buffer: B,
    pub transient: Option<Buffer>,
}

#[derive(Debug)]
pub struct BufferDescriptor<'a> {
    data: DescriptorData<'a>,
    mutability: BufferMutability,
    n_elems: u32,
    buffer_type: BufferTypeDesc,
}

pub trait IntoBufferData {
    fn data(&self) -> DescriptorData<'_>;
    fn n_elems(&self) -> u32;
    fn buffer_type(&self) -> BufferTypeDesc;
}

pub trait BufferContents {}

impl<T> IntoBufferData for [T]
where
    T: BufferContents,
{
    fn data(&self) -> DescriptorData<'_> {
        todo!()
    }

    fn buffer_type(&self) -> BufferTypeDesc {
        todo!()
    }

    fn n_elems(&self) -> u32 {
        todo!()
    }
}

impl<'a> BufferDescriptor<'a> {
    pub fn from_slice<D: IntoBufferData>(data: &D, mutability: BufferMutability) -> Self {
        todo!()
    }
}

/*
TODO: Probably needs to live on the HostBuffer instead
impl<BT: Clone + BufferType> BufferDescriptor<'static> {
    pub fn from_host_buffer(hb: &HostBuffer<BT>, mutability: BufferMutability) -> Self {
        Self {
            data: DescriptorData::Shared(hb.data.clone()),
            mutability,
            n_elems: hb.n_elems,
            buffer_type: hb.buffer_type.clone(),
        }
    }
}

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

//impl_descriptor_from!(BufferDescriptor, Uniform, UniformBufferType);
//impl_descriptor_from!(BufferDescriptor, VertexDefinition, VertexBufferType);
//impl_descriptor_from!(BufferDescriptor, IndexInt, IndexBufferType);
// impl_descriptor_from!(BufferDescriptor, Std140, StorageBufferType);

 */

impl<'a> BufferDescriptor<'a> {
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

impl<'a> BufferDescriptor<'a> {
    pub fn vk_usage_flags(&self) -> vk::BufferUsageFlags {
        todo!()
    }

    pub fn elem_size(&self) -> u16 {
        self.buffer_type.elem_size()
    }

    pub fn elem_align(&self, allocator: &AllocatorHandle) -> u16 {
        self.buffer_type.elem_align(allocator)
    }

    pub fn buffer_type(&self) -> &BufferTypeDesc {
        &self.buffer_type
    }
}

impl<'a> BufferDescriptor<'a> {
    pub fn enqueue_single(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<BufferResult<DeviceBuffer>, MemoryError> {
        let (buffer, transient) = DeviceBuffer::create(allocator, command_buffer, self)?;

        Ok(BufferResult { buffer, transient })
    }

    #[allow(clippy::type_complexity)]
    pub fn enqueue(
        &self,
        allocator: &AllocatorHandle,
        command_buffer: &mut CommandBuffer,
    ) -> Result<
        (
            BufferResult<DeviceBuffer>,
            Option<BufferResult<DeviceBuffer>>,
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
