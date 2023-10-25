pub use resurs::*;

use crate::buffer;
use crate::pipeline;
use crate::pipeline_resource;
use crate::render_pass;
use crate::render_target;
use crate::texture;

#[derive(Default)]
pub struct AsyncResources {
    pub uniform_buffers: buffer::AsyncUniformBuffers,
    pub vertex_buffers: buffer::AsyncVertexBuffers,
    pub index_buffers: buffer::AsyncIndexBuffers,
    pub textures: texture::AsyncTextures,
}

pub struct Resources {
    pub uniform_buffers: buffer::UniformBuffers,
    pub vertex_buffers: buffer::VertexBuffers,
    pub index_buffers: buffer::IndexBuffers,
    pub textures: texture::Textures,
    pub graphics_pipelines: pipeline::GraphicsPipelines,
    pub descriptor_sets: pipeline_resource::PipelineResourceSetStorage,
    pub render_passes: resurs::Storage<render_pass::RenderPass>,
    pub render_targets: resurs::Storage<render_target::RenderTarget>,
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
