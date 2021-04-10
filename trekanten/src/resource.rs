pub use resurs::*;

use crate::descriptor;
use crate::mem;
use crate::pipeline;
use crate::render_pass;
use crate::texture;

pub enum ResourceCommand {
    CreateVertexBuffer {
        descriptor: mem::OwningVertexBufferDescriptor,
        handle: mem::BufferHandle<mem::VertexBuffer>,
    },
    CreateIndexBuffer {
        descriptor: mem::OwningIndexBufferDescriptor,
        handle: mem::BufferHandle<mem::IndexBuffer>,
    },
    CreateUniformBuffer {
        descriptor: mem::OwningUniformBufferDescriptor,
        handle: mem::BufferHandle<mem::UniformBuffer>,
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

pub struct ResourceCommandBatch {
    _commands: arrayvec::ArrayVec<[ResourceCommand; 64]>,
}

#[derive(Default)]
pub struct AsyncResources {
    pub uniform_buffers: mem::AsyncUniformBuffers,
    pub vertex_buffers: mem::AsyncVertexBuffers,
    pub index_buffers: mem::AsyncIndexBuffers,
    pub textures: texture::AsyncTextures,
    pub graphics_pipelines: pipeline::AsyncGraphicsPipelines,
}

pub struct Resources {
    pub uniform_buffers: mem::UniformBuffers,
    pub vertex_buffers: mem::VertexBuffers,
    pub index_buffers: mem::IndexBuffers,
    pub textures: texture::Textures,
    pub graphics_pipelines: pipeline::GraphicsPipelines,
    pub descriptor_sets: descriptor::DescriptorSets,
    pub render_passes: resurs::Storage<render_pass::RenderPass>,
}

pub trait ResourceManager<Descriptor, Resource, Handle> {
    type Error;
    fn get_resource(&self, handle: &Handle) -> Option<&Resource>;
    fn create_resource_blocking(&mut self, descriptor: Descriptor) -> Result<Handle, Self::Error>;
}

pub trait MutResourceManager<Descriptor, Resource, Handle> {
    type Error;
    fn recreate_resource_blocking(
        &mut self,
        handle: Handle,
        descriptor: Descriptor,
    ) -> Result<Handle, Self::Error>;
}
