use crate::ecs::prelude::*;
use crate::render::GpuBuffer;

use trekant::loader::Loader;
use trekant::{AsyncBufferHandle, BufferDescriptor, BufferHandle, BufferMutability};
use trekant::{HostIndexBuffer, HostVertexBuffer};

use ram_derive::Visitable;

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
        let vbuf_desc = BufferDescriptor::from_host_buffer(
            &self.cpu_vertex_buffer,
            BufferMutability::Immutable,
            None,
        );
        let ibuf_desc = BufferDescriptor::from_host_buffer(
            &self.cpu_index_buffer,
            BufferMutability::Immutable,
            None,
        );

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
