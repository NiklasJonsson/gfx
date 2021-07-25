use crate::ecs::prelude::*;
use crate::render::Pending;

use ramneryd_derive::Inspect;
use trekanten::buffer::{DeviceIndexBuffer, DeviceVertexBuffer, HostIndexBuffer, HostVertexBuffer};
use trekanten::loader::{Loader, ResourceLoader};
use trekanten::resource::Async;
use trekanten::{BufferHandle, BufferMutability};

/// Utility for working with asynchronously uploaded gpu resources
#[derive(Inspect)]
pub enum GpuResource<InFlightT, AvailT> {
    Null,
    InFlight(InFlightT),
    Available(AvailT),
}
type GpuBuffer<BT> = GpuResource<BufferHandle<Async<BT>>, BufferHandle<BT>>;

#[derive(Component)]
#[component(inspect)]
pub struct Mesh {
    pub cpu_vertex_buffer: HostVertexBuffer,
    pub cpu_index_buffer: HostIndexBuffer,
    pub gpu_vertex_buffer: GpuBuffer<DeviceVertexBuffer>,
    pub gpu_index_buffer: GpuBuffer<DeviceIndexBuffer>,
}

#[derive(Component)]
#[component(inspect)]
pub struct GpuMesh {
    pub vertex_buffer: BufferHandle<DeviceVertexBuffer>,
    pub index_buffer: BufferHandle<DeviceIndexBuffer>,
    pub polygon_mode: trekanten::pipeline::PolygonMode,
}

#[derive(Component, Clone)]
#[component(inspect)]
pub struct CpuMesh {
    pub vertex_buffer: HostVertexBuffer,
    pub index_buffer: HostIndexBuffer,
    pub polygon_mode: trekanten::pipeline::PolygonMode,
}

#[derive(Component)]
#[component(inspect)]
pub struct PendingMesh {
    pub vertex_buffer:
        Pending<BufferHandle<Async<DeviceVertexBuffer>>, BufferHandle<DeviceVertexBuffer>>,
    pub index_buffer:
        Pending<BufferHandle<Async<DeviceIndexBuffer>>, BufferHandle<DeviceIndexBuffer>>,
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
        use trekanten::buffer::{IndexBufferDescriptor, VertexBufferDescriptor};
        let vbuf_desc = VertexBufferDescriptor::from_host_buffer(
            &mesh.vertex_buffer,
            BufferMutability::Immutable,
        );
        let ibuf_desc = IndexBufferDescriptor::from_host_buffer(
            &mesh.index_buffer,
            BufferMutability::Immutable,
        );
        Self {
            vertex_buffer: Pending::Pending(
                loader
                    .load(vbuf_desc)
                    .expect("Failed to load vertex buffer"),
            ),
            index_buffer: Pending::Pending(
                loader.load(ibuf_desc).expect("Failed to load index buffer"),
            ),
            polygon_mode: mesh.polygon_mode,
        }
    }
}
