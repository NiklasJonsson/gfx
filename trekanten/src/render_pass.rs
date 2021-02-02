use crate::mem::{BufferHandle, IndexBuffer, VertexBuffer};
use crate::resource::Handle;
use crate::util;

use crate::backend::vk::command::{CommandBuffer, CommandError};
use crate::descriptor::DescriptorSet;
use crate::pipeline::{GraphicsPipeline, ShaderStage};
use crate::resource::Resources;

pub struct RenderPassEncoder<'a> {
    resources: &'a Resources,
    frame_idx: u32,
    command_buffer: CommandBuffer,
}

impl<'a> RenderPassEncoder<'a> {
    pub fn bind_shader_resource_group(
        &mut self,
        idx: u32,
        dset: &Handle<DescriptorSet>,
        pipeline: &Handle<GraphicsPipeline>,
    ) -> &mut Self {
        let dset = self
            .resources
            .descriptor_sets
            .get(dset, self.frame_idx as usize)
            .expect("Failed to find descriptor set");

        let pipeline = self
            .resources
            .graphics_pipelines
            .get(&pipeline)
            .expect("Failed to find pipeline");

        self.command_buffer.bind_descriptor_set(idx, dset, pipeline);

        self
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &Handle<GraphicsPipeline>) -> &mut Self {
        let pipeline = self
            .resources
            .graphics_pipelines
            .get(&pipeline)
            .expect("Failed to get pipeline");

        self.command_buffer.bind_graphics_pipeline(pipeline);

        self
    }

    pub fn bind_index_buffer(&mut self, handle: &BufferHandle<IndexBuffer>) -> &mut Self {
        let ib = self
            .resources
            .index_buffers
            .get(&handle, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        self.command_buffer.bind_index_buffer(&ib, 0);

        self
    }

    pub fn bind_vertex_buffer(&mut self, handle: &BufferHandle<VertexBuffer>) -> &mut Self {
        let vb = self
            .resources
            .vertex_buffers
            .get(&handle, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        self.command_buffer.bind_vertex_buffer(&vb, 0);

        self
    }

    pub fn draw_mesh(
        &mut self,
        vertex_buffer: &BufferHandle<VertexBuffer>,
        index_buffer: &BufferHandle<IndexBuffer>,
    ) -> &mut Self {
        let vertex_index = vertex_buffer.idx() as i32;
        let indices_index = index_buffer.idx();
        let n_indices = index_buffer.n_elems();

        let vb = self
            .resources
            .vertex_buffers
            .get(&vertex_buffer, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        let ib = self
            .resources
            .index_buffers
            .get(&index_buffer, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        self.command_buffer
            .bind_index_buffer(ib, 0)
            .bind_vertex_buffer(vb, 0)
            .draw_indexed(n_indices, indices_index, vertex_index);

        self
    }

    pub fn draw_indexed(
        &mut self,
        n_indices: u32,
        indices_index: u32,
        vertices_index: i32,
    ) -> &mut Self {
        self.command_buffer
            .draw_indexed(n_indices, indices_index, vertices_index);

        self
    }

    pub fn set_scissor(&mut self, scissor: util::Rect2D) -> &mut Self {
        self.command_buffer.set_scissor(scissor);

        self
    }

    pub fn set_viewport(&mut self, viewport: util::Viewport) -> &mut Self {
        self.command_buffer.set_viewport(viewport);

        self
    }

    pub fn bind_push_constant<V: Copy>(
        &mut self,
        pipeline: &Handle<GraphicsPipeline>,
        stage: ShaderStage,
        v: &V,
    ) -> &mut Self {
        let pipeline = self
            .resources
            .graphics_pipelines
            .get(&pipeline)
            .expect("Failed to get pipeline");

        self.command_buffer.bind_push_constant(pipeline, stage, v);

        self
    }

    pub fn new(resources: &'a Resources, command_buffer: CommandBuffer, frame_idx: u32) -> Self {
        Self {
            resources,
            command_buffer,
            frame_idx,
        }
    }

    pub fn build(mut self) -> Result<CommandBuffer, CommandError> {
        self.command_buffer.end_render_pass().end()?;
        Ok(self.command_buffer)
    }

    pub fn inner(self) -> CommandBuffer {
        let Self { command_buffer, .. } = self;
        command_buffer
    }
}
