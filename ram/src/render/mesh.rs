use crate::ecs::prelude::*;
use crate::render::GpuBuffer;

use trekant::buffer::{DeviceIndexBuffer, DeviceVertexBuffer, HostIndexBuffer, HostVertexBuffer};
use trekant::loader::{Loader, ResourceLoader};
use trekant::resource::Async;
use trekant::{BufferHandle, BufferMutability};

use ram_derive::Visitable;

#[derive(Component, Visitable)]
pub struct Mesh {
    pub cpu_vertex_buffer: HostVertexBuffer,
    pub cpu_index_buffer: HostIndexBuffer,
    pub gpu_vertex_buffer: GpuBuffer<DeviceVertexBuffer>,
    pub gpu_index_buffer: GpuBuffer<DeviceIndexBuffer>,
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

fn map_buffer_handle<BT>(
    h: &mut GpuBuffer<BT>,
    old: BufferHandle<Async<BT>>,
    new: BufferHandle<BT>,
) -> bool {
    match h {
        GpuBuffer::InFlight(cur) if cur.handle() == old.handle() => {
            *h = GpuBuffer::Available(BufferHandle::sub_buffer(new, cur.idx(), cur.n_elems()));
            true
        }
        _ => false,
    }
}

impl Mesh {
    pub fn try_consume_vertex_buffer(
        &mut self,
        old: BufferHandle<Async<DeviceVertexBuffer>>,
        new: BufferHandle<DeviceVertexBuffer>,
    ) -> bool {
        map_buffer_handle(&mut self.gpu_vertex_buffer, old, new)
    }

    pub fn try_consume_index_buffer(
        &mut self,
        old: BufferHandle<Async<DeviceIndexBuffer>>,
        new: BufferHandle<DeviceIndexBuffer>,
    ) -> bool {
        map_buffer_handle(&mut self.gpu_index_buffer, old, new)
    }

    pub fn load_gpu(&mut self, loader: &Loader) {
        use trekant::buffer::{IndexBufferDescriptor, VertexBufferDescriptor};
        let vbuf_desc = VertexBufferDescriptor::from_host_buffer(
            &self.cpu_vertex_buffer,
            BufferMutability::Immutable,
        );
        let ibuf_desc = IndexBufferDescriptor::from_host_buffer(
            &self.cpu_index_buffer,
            BufferMutability::Immutable,
        );

        self.gpu_vertex_buffer = GpuBuffer::InFlight(
            loader
                .load(vbuf_desc)
                .expect("Failed to load vertex buffer"),
        );
        self.gpu_index_buffer =
            GpuBuffer::InFlight(loader.load(ibuf_desc).expect("Failed to load index buffer"));
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
