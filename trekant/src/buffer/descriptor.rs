use super::{
    BufferMutability, BufferType, DeviceBuffer, IndexBufferType, IndexInt, StorageBufferType,
    UniformBufferType, VertexBufferType,
};
use crate::vertex::VertexDefinition;
use crate::{backend, Std140};

use crate::util::ByteBuffer;
use backend::buffer::Buffer;
use backend::command::CommandBuffer;
use backend::{AllocatorHandle, MemoryError};

use crate::descriptor::DescriptorData;
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
    buffer_type: BufferType,
}

// TODO: Super-trait of Deref<Target = [T]>?
pub trait BufferData<'a> {
    type T;
    fn n_elems(&self) -> u32;
    fn to_byte_data(self) -> DescriptorData<'a>;
}

impl<T> BufferData<'static> for Vec<T>
where
    T: bytemuck::Pod,
{
    type T = T;

    fn n_elems(&self) -> u32 {
        self.len()
            .try_into()
            .expect("Vec length does not fit in u64")
    }

    fn to_byte_data(self) -> DescriptorData<'static> {
        let value = unsafe { ByteBuffer::from_vec(self) };
        DescriptorData::Owned(value)
    }
}

impl<'a, T> BufferData<'a> for &'a [T]
where
    T: bytemuck::Pod,
{
    type T = T;
    fn n_elems(&self) -> u32 {
        self.len()
            .try_into()
            .expect("Span length does not fit into u64")
    }

    fn to_byte_data(self) -> DescriptorData<'a> {
        let value: &[u8] = bytemuck::cast_slice(self);
        DescriptorData::Borrowed(value)
    }
}

// The impls for &Vec and &[T, N] are only for convenience so that the user doesn't have to call e.g. as_slice()

/// This impl ensures that we can call any_buffer with a reference to a vector. This is treated the same as
/// if the caller was passing a slice. It seems that only the slice implementation is not enough for this to
/// work due to how type inference works.
impl<'a, T> BufferData<'a> for &'a Vec<T>
where
    T: bytemuck::Pod,
{
    type T = T;
    fn n_elems(&self) -> u32 {
        self.len()
            .try_into()
            .expect("Span length does not fit into u64")
    }

    fn to_byte_data(self) -> DescriptorData<'a> {
        self.as_slice().to_byte_data()
    }
}

impl<'a, T> BufferData<'a> for &'a mut Vec<T>
where
    T: bytemuck::Pod,
{
    type T = T;
    fn n_elems(&self) -> u32 {
        self.len()
            .try_into()
            .expect("Span length does not fit into u64")
    }

    fn to_byte_data(self) -> DescriptorData<'a> {
        self.as_slice().to_byte_data()
    }
}

/// Same as the &Vec<T> impl, this is a convenience feature.
impl<'a, T, const N: usize> BufferData<'a> for &'a [T; N]
where
    T: bytemuck::Pod,
{
    type T = T;
    fn n_elems(&self) -> u32 {
        // TODO: Can we get a compile time bound here?
        N.try_into().expect("Doesn't fit")
    }

    fn to_byte_data(self) -> DescriptorData<'a> {
        self.as_slice().to_byte_data()
    }
}

/// Defines how individual elements within the buffer should be aligned.
/// If the buffer will have individual elements bounds to a descriptor set,
/// use MinBufferOffset. If the whole buffer is bound in a shader, use Std140.
#[derive(Debug, Clone, Copy)]
pub enum BufferLayout {
    MinBufferOffset,
    Std140,
}

fn compute_required_alignment(bl: BufferLayout, std140_element_alignment: u16) -> u16 {
    match bl {
        BufferLayout::MinBufferOffset => backend::buffer::MIN_BUFFER_OFFSET,
        BufferLayout::Std140 => std140_element_alignment,
    }
}

/// The decision of what to make unsafe is a bit tricky. If the buffer is bound to a descriptor set
/// and it is not compatible with the pipeline that descriptor set is bound to, then that is invalid usage
/// of Vulkan (unsure if it is technically UB) but that is pretty much impossible to enforce in the buffer
/// API here. Instead, the unsafety of these functions depend on if the type of the data itself is possible
/// to use on the GPU. These are general alignment requirements etc. that hold for the data layout inside the
/// buffer. How the buffer is going to be used has some effect on this, e.g. there are different requirements
/// for vertex data and uniform data. This is why type traits are introduced to capture the purpose of the type
/// and these guarantee that the data inside the buffer is proper if a buffer is created with this buffer descriptor.
/// Still though, it is up to the user to correctly use this buffer with the rest of the vulkan API.
impl<'a> BufferDescriptor<'a> {
    /// # Safety
    ///
    /// The byte buffer in 'descriptor_data' needs to be compatible with how it is used with the vulkan API, e.g. as a uniform buffer or a vertex buffer.
    pub unsafe fn raw_buffer(
        descriptor_data: DescriptorData<'a>,
        mutability: BufferMutability,
        buffer_type: BufferType,
        n_elems: u32,
    ) -> Self {
        Self {
            data: descriptor_data,
            mutability,
            n_elems,
            buffer_type,
        }
    }

    /// # Safety
    ///
    /// The buffer data needs to be compatible with how it is used with the vulkan API, e.g. as a uniform buffer or a vertex buffer.
    pub unsafe fn any_buffer<BD>(
        data: BD,
        mutability: BufferMutability,
        buffer_type: BufferType,
    ) -> Self
    where
        BD: BufferData<'a>,
        BD::T: bytemuck::Pod,
    {
        let n_elems = data.n_elems();
        let descriptor_data = data.to_byte_data();
        // Safety: The bytemuck::Pod trait ensures this is "convertible to bytes." and usable on the GPU.
        unsafe { Self::raw_buffer(descriptor_data, mutability, buffer_type, n_elems) }
    }

    pub fn index_buffer<BD>(data: BD, mutability: BufferMutability) -> Self
    where
        BD: BufferData<'a>,
        BD::T: IndexInt + bytemuck::Pod,
    {
        let buffer_type = IndexBufferType::as_enum::<BD::T>();
        // Safety: This is safe because of the IndexInt
        unsafe { Self::any_buffer(data, mutability, buffer_type) }
    }

    pub fn vertex_buffer<BD>(data: BD, mutability: BufferMutability) -> Self
    where
        BD: BufferData<'a>,
        BD::T: VertexDefinition + bytemuck::Pod,
    {
        let buffer_type = VertexBufferType::as_enum::<BD::T>();
        // Safety: The VertexDefinition trait
        unsafe { Self::any_buffer(data, mutability, buffer_type) }
    }

    pub fn uniform_buffer<BD>(
        data: BD,
        mutability: BufferMutability,
        buffer_layout: BufferLayout,
    ) -> Self
    where
        BD: BufferData<'a>,
        BD::T: Std140 + bytemuck::Pod,
    {
        let elem_align = compute_required_alignment(
            buffer_layout,
            <BD::T as Std140>::ALIGNMENT.try_into().unwrap(),
        );
        let elem_size = <BD::T as Std140>::SIZE.try_into().unwrap();
        let buffer_type = BufferType::Uniform(UniformBufferType {
            elem_size,
            elem_align,
        });
        // Safety: The Std140 trait
        unsafe { Self::any_buffer(data, mutability, buffer_type) }
    }

    pub fn storage_buffer<BD>(data: BD, mutability: BufferMutability) -> Self
    where
        BD: BufferData<'a>,
        BD::T: bytemuck::Pod + Std140,
    {
        let buffer_type = StorageBufferType::as_enum::<BD::T>();
        // Safety: Std140 trait
        unsafe { Self::any_buffer(data, mutability, buffer_type) }
    }
}

impl<'a> BufferDescriptor<'a> {
    pub fn mutability(&self) -> BufferMutability {
        self.mutability
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }

    pub fn is_empty(&self) -> bool {
        self.n_elems == 0
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
        match self.buffer_type {
            BufferType::Index(_) => vk::BufferUsageFlags::INDEX_BUFFER,
            BufferType::Vertex(_) => vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferType::Storage(_) => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferType::Uniform(_) => vk::BufferUsageFlags::UNIFORM_BUFFER,
        }
    }

    pub fn elem_size(&self) -> u16 {
        self.buffer_type.elem_size()
    }

    pub fn elem_align(&self) -> u16 {
        self.buffer_type.elem_align()
    }

    pub fn buffer_type(&self) -> &BufferType {
        &self.buffer_type
    }
}

// TODO: Revisit this. Should this really be a method on this class? Look at the texture code for inspo.
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

#[cfg(test)]
mod tests {
    use crate::Format;

    use super::*;

    #[test]
    fn index_buffer() {
        let v32: Vec<u32> = vec![10u32, 11u32, 100u32];
        let v16: Vec<u16> = vec![10u16, 11u16, 100u16];
        let v32_2: Vec<u32> = v32.clone();
        let v16_2: Vec<u16> = v16.clone();
        let descriptors = [
            BufferDescriptor::index_buffer(v16, BufferMutability::Immutable),
            BufferDescriptor::index_buffer(v32, BufferMutability::Mutable),
            BufferDescriptor::index_buffer(&v16_2, BufferMutability::Mutable),
            BufferDescriptor::index_buffer(&v32_2, BufferMutability::Mutable),
        ];

        for (i, d) in descriptors.into_iter().enumerate() {
            assert_eq!(d.n_elems(), 3);
            assert_eq!(d.elem_size(), if i % 2 == 0 { 2 } else { 4 });
        }
    }

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    #[repr(C)]
    struct Vertex {
        x: [f32; 4],
        y: f32,
    }
    unsafe impl VertexDefinition for Vertex {
        fn format() -> crate::VertexFormat {
            crate::VertexFormat::from([Format::FLOAT4, Format::FLOAT1])
        }
    }

    #[test]
    fn vertex_buffer() {
        let v0: Vec<Vertex> = vec![
            Vertex {
                x: [0.0; 4],
                y: 10.0,
            },
            Vertex {
                x: [0.0; 4],
                y: 10.0,
            },
        ];
        let v1 = v0.clone();
        let descriptors = [
            BufferDescriptor::vertex_buffer(v0, BufferMutability::Immutable),
            BufferDescriptor::vertex_buffer(&v1, BufferMutability::Mutable),
        ];

        for d in descriptors {
            assert_eq!(d.n_elems(), 2);
            assert_eq!(d.elem_size() as usize, std::mem::size_of::<f32>() * 5);
        }
    }
}
