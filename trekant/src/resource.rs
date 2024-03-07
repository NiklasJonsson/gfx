pub use resurs::*;

use crate::buffer;
use crate::pipeline;
use crate::pipeline_resource;
use crate::render_pass;
use crate::render_target;
use crate::texture;

pub struct Resources {
    pub buffers: buffer::Buffers,
    pub textures: texture::Textures,
    pub graphics_pipelines: pipeline::GraphicsPipelines,
    pub descriptor_sets: pipeline_resource::PipelineResourceSetStorage,
    pub render_passes: resurs::Storage<render_pass::RenderPass>,
    pub render_targets: resurs::Storage<render_target::RenderTarget>,
}
