use crate::ecs::prelude::*;
use crate::render::Pending;
use trekanten::buffer::{
    IndexBuffer, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor, VertexBuffer,
};
use trekanten::loader::{Loader, ResourceLoader};
use trekanten::resource::Async;
use trekanten::util::ByteBuffer;
use trekanten::BufferHandle;

pub enum GpuResource<PendingT, AvailT> {
    Null,
    Pending(PendingT),
    Available(AvailT),
}
type GpuBuffer<BT> = GpuResource<BufferHandle<Async<BT>>, BufferHandle<BT>>;

pub struct Mesh {
    pub cpu_vertex_buffer: ByteBuffer,
    pub cpu_index_buffer: ByteBuffer,
    pub gpu_vertex_buffer: GpuBuffer<VertexBuffer>,
    pub gpu_index_buffer: GpuBuffer<IndexBuffer>,
}

#[derive(Component)]
#[component(inspect)]
pub struct GpuMesh {
    pub vertex_buffer: BufferHandle<VertexBuffer>,
    pub index_buffer: BufferHandle<IndexBuffer>,
    pub polygon_mode: trekanten::pipeline::PolygonMode,
}

#[derive(Component, Clone)]
#[component(inspect)]
pub struct CpuMesh {
    pub vertex_buffer: OwningVertexBufferDescriptor,
    pub index_buffer: OwningIndexBufferDescriptor,
    pub polygon_mode: trekanten::pipeline::PolygonMode,
}

#[derive(Component)]
#[component(inspect)]
pub struct PendingMesh {
    pub vertex_buffer: Pending<BufferHandle<Async<VertexBuffer>>, BufferHandle<VertexBuffer>>,
    pub index_buffer: Pending<BufferHandle<Async<IndexBuffer>>, BufferHandle<IndexBuffer>>,
    pub polygon_mode: trekanten::pipeline::PolygonMode,
}

impl PendingMesh {
    pub fn try_finish(&self) -> Option<GpuMesh> {
        match (&self.vertex_buffer, &self.index_buffer) {
            (Pending::Available(vb), Pending::Available(ib)) => Some(GpuMesh {
                vertex_buffer: vb.clone(),
                index_buffer: ib.clone(),
                polygon_mode: self.polygon_mode,
            }),
            _ => None,
        }
    }

    pub fn load(loader: &Loader, mesh: &CpuMesh) -> Self {
        Self {
            vertex_buffer: Pending::Pending(
                loader
                    .load(mesh.vertex_buffer.clone())
                    .expect("Failed to load vertex buffer"),
            ),
            index_buffer: Pending::Pending(
                loader
                    .load(mesh.index_buffer.clone())
                    .expect("Failed to load index buffer"),
            ),
            polygon_mode: mesh.polygon_mode,
        }
    }
}
