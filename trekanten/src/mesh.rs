use crate::mem::{BufferHandle, IndexBuffer, VertexBuffer};

pub struct Mesh {
    pub vertex_buffer: BufferHandle<VertexBuffer>,
    pub index_buffer: BufferHandle<IndexBuffer>,
}
