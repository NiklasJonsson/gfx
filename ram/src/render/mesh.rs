use crate::ecs::prelude::*;
use crate::render::GpuBuffer;

use trekant::ByteBuffer;
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

// TODO: Split into Pending/Done etc.?
#[derive(Component, Visitable)]
pub struct Mesh {
    pub cpu_vertex_buffer: HostVertexBuffer,
    pub cpu_index_buffer: HostIndexBuffer,
    pub gpu_vertex_buffer: Option<GpuBuffer>,
    pub gpu_index_buffer: Option<GpuBuffer>,
}

impl Mesh {
    pub fn new(vertex_buffer: HostVertexBuffer, index_buffer: HostIndexBuffer) -> Self {
        Self {
            cpu_vertex_buffer: vertex_buffer,
            cpu_index_buffer: index_buffer,
            gpu_vertex_buffer: None,
            gpu_index_buffer: None,
        }
    }
}

struct MeshLoad;
impl MeshLoad {
    pub const ID: &'static str = "MeshLoad";
}

const MESH_LOAD_ID: trekant::LoadId = trekant::LoadId("MeshLoad");

#[derive(Default)]
struct PendingBuffers(crate::render::PendingEntityBuffers);

impl<'a> System<'a> for MeshLoad {
    type SystemData = (
        WriteExpect<'a, trekant::Loader>,
        WriteStorage<'a, Mesh>,
        Write<'a, PendingBuffers>,
        Entities<'a>,
    );

    fn run(&mut self, (loader, mut meshes, mut pending_buffers, entities): Self::SystemData) {
        for (entity, mesh) in (&entities, &mut meshes).join() {
            let should_load = mesh.gpu_index_buffer.is_none() && mesh.gpu_vertex_buffer.is_none();
            if should_load {
                let vbuf_desc =
                    host_buffer_to_desc(&mesh.cpu_vertex_buffer, BufferMutability::Immutable);
                let ibuf_desc =
                    host_buffer_to_desc(&mesh.cpu_index_buffer, BufferMutability::Immutable);

                let vbuf = loader
                    .load_buffer(vbuf_desc, MESH_LOAD_ID)
                    .expect("Failed to load vertex buffer");
                let ibuf = loader
                    .load_buffer(ibuf_desc, MESH_LOAD_ID)
                    .expect("Failed to load index buffer");

                for buf in [ibuf, vbuf] {
                    pending_buffers.0.push(buf, entity);
                }
                mesh.gpu_vertex_buffer = Some(GpuBuffer::Pending(vbuf));
                mesh.gpu_index_buffer = Some(GpuBuffer::Pending(ibuf));
            }
        }

        use trekant::HandleMapping;
        for mapping in loader.flush(MESH_LOAD_ID) {
            let HandleMapping::Buffer { old, new } = mapping else {
                panic!("Mesh pipeline has no textures");
            };
            for entity in pending_buffers.0.flush(old) {
                let Some(mesh) = meshes.get_mut(entity) else {
                    log::debug!("Entity {entity:?} had a pending buffer load for mesh data but was destroyed while the buffer was loading");
                    continue;
                };

                for buf in [&mut mesh.gpu_index_buffer, &mut mesh.gpu_vertex_buffer] {
                    buf.as_mut().unwrap().try_take(old, new);
                }
            }
        }
    }
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    builder.with(MeshLoad, MeshLoad::ID, &[])
}
