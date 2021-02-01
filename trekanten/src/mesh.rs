use crate::mem::{IndexBuffer, VertexBuffer, BufferHandle};

pub struct Mesh {
    pub vertex_buffer: BufferHandle<VertexBuffer>,
    pub index_buffer: BufferHandle<IndexBuffer>,
}
