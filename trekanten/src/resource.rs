pub use resurs::*;

use crate::mem;
use crate::mesh;
use crate::pipeline;
use crate::texture;
use crate::uniform;

pub enum ResourceCommand {
    CreateVertexBuffer {
        descriptor: mesh::OwningVertexBufferDescriptor,
        handle: mem::BufferHandle<mesh::VertexBuffer>,
    },
    CreateIndexBuffer {
        descriptor: mesh::OwningIndexBufferDescriptor,
        handle: mem::BufferHandle<mesh::IndexBuffer>,
    },
    CreateUniformBuffer {
        descriptor: uniform::OwningUniformBufferDescriptor,
        handle: mem::BufferHandle<uniform::UniformBuffer>,
    },
    CreateTexture {
        descriptor: texture::TextureDescriptor,
        handle: Handle<texture::Texture>,
    },
    CreatePipeline {
        descriptor: pipeline::GraphicsPipelineDescriptor,
        handle: Handle<pipeline::GraphicsPipeline>,
    },
}

#[derive(Default)]
pub struct AsyncResources {
    pub uniform_buffers: uniform::AsyncUniformBuffers,
    pub vertex_buffers: mesh::AsyncVertexBuffers,
    pub index_buffers: mesh::AsyncIndexBuffers,
    pub textures: texture::AsyncTextures,
    pub graphics_pipelines: pipeline::AsyncGraphicsPipelines,
}

use parking_lot::MappedRwLockReadGuard;
pub trait ResourceManager<Descriptor, Resource, Handle> {
    type Error;
    fn get_resource(&self, handle: &Handle) -> Option<MappedRwLockReadGuard<'_, Async<Resource>>>;
    fn create_resource_blocking(&mut self, descriptor: Descriptor) -> Result<Handle, Self::Error>;
}

pub trait MutResourceManager<Descriptor, Resource, Handle> {
    type Error;
    fn get_resource_mut(&mut self, handle: &Handle) -> Option<&mut Resource>;
    fn recreate_resource(
        &mut self,
        handle: Handle,
        descriptor: Descriptor,
    ) -> Result<Handle, Self::Error>;
}
