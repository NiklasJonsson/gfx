use crate::ecs::prelude::*;
use crate::render::Pending;
use trekanten::loader::{Loader, ResourceLoader};
use trekanten::mem::{IndexBuffer, VertexBuffer};
use trekanten::resource::Async;
use trekanten::BufferHandle;

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
    pub vertex_buffer: trekanten::mem::OwningVertexBufferDescriptor,
    pub index_buffer: trekanten::mem::OwningIndexBufferDescriptor,
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
