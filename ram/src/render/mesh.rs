use crate::ecs::prelude::*;
use crate::render::GpuBuffer;

use trekant::ByteBuffer;
use trekant::Loader;
use trekant::{
    BufferDescriptor, BufferMutability, IndexBufferType, IndexInt, VertexBufferType,
    VertexDefinition, VertexFormat,
};

use trekant::BufferTypeTrait as _;

use ram_derive::Visitable;

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct HostBuffer<BT> {
    data: Arc<ByteBuffer>,
    n_elems: u32,
    buffer_type: BT,
}

pub type HostVertexBuffer = HostBuffer<VertexBufferType>;
pub type HostIndexBuffer = HostBuffer<IndexBufferType>;

macro_rules! impl_host_buffer_from {
    ($name:ty, $trait:ident, $buffer_type:ident) => {
        impl $name {
            pub fn from_vec<T: Copy + $trait + 'static>(data: Vec<T>) -> Self {
                let n_elems = data.len() as u32;
                let data = Arc::new(unsafe { ByteBuffer::from_vec(data) });
                let buffer_type = $buffer_type::from_type::<T>();
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

impl_host_buffer_from!(HostVertexBuffer, VertexDefinition, VertexBufferType);
impl_host_buffer_from!(HostIndexBuffer, IndexInt, IndexBufferType);

impl HostVertexBuffer {
    pub fn format(&self) -> &VertexFormat {
        self.buffer_type.format()
    }
}

pub fn host_buffer_to_desc<BT>(
    hb: &HostBuffer<BT>,
    mutability: BufferMutability,
) -> BufferDescriptor<'static>
where
    BT: trekant::BufferTypeTrait + Clone,
{
    let buffer_type = hb.buffer_type.buffer_type();
    let data = trekant::DescriptorData::Shared(std::sync::Arc::clone(&hb.data));

    // SAFETY: We use the same traits as trekant here which guarantee safety.
    unsafe { BufferDescriptor::raw_buffer(data, mutability, buffer_type, hb.n_elems) }
}

#[derive(Component, Visitable)]
pub struct Mesh {
    pub cpu_vertex_buffer: HostVertexBuffer,
    pub cpu_index_buffer: HostIndexBuffer,
    pub gpu_vertex_buffer: GpuBuffer,
    pub gpu_index_buffer: GpuBuffer,
}

impl Mesh {
    pub fn new(vertex_buffer: HostVertexBuffer, index_buffer: HostIndexBuffer) -> Self {
        Self {
            cpu_vertex_buffer: vertex_buffer,
            cpu_index_buffer: index_buffer,
            gpu_vertex_buffer: GpuBuffer::None,
            gpu_index_buffer: GpuBuffer::None,
        }
    }
}

impl Mesh {
    pub fn load_gpu(&mut self, loader: &Loader) {
        let vbuf_desc = host_buffer_to_desc(&self.cpu_vertex_buffer, BufferMutability::Immutable);
        let ibuf_desc = host_buffer_to_desc(&self.cpu_index_buffer, BufferMutability::Immutable);

        self.gpu_vertex_buffer = GpuBuffer::InFlight(
            loader
                .load_buffer(vbuf_desc)
                .expect("Failed to load vertex buffer"),
        );
        self.gpu_index_buffer = GpuBuffer::InFlight(
            loader
                .load_buffer(ibuf_desc)
                .expect("Failed to load index buffer"),
        );
    }
}

impl Mesh {
    pub fn is_available_gpu(&self) -> bool {
        std::matches!(
            (&self.gpu_vertex_buffer, &self.gpu_index_buffer),
            (GpuBuffer::Available(_), GpuBuffer::Available(_))
        )
    }

    pub fn is_pending_gpu(&self) -> bool {
        std::matches!(&self.gpu_vertex_buffer, GpuBuffer::InFlight(_))
            || std::matches!(&self.gpu_index_buffer, GpuBuffer::InFlight(_))
    }
}
