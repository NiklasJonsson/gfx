use crate::mem::{BufferHandle, BufferStorageReadGuard, IndexBuffer, VertexBuffer};
use crate::mesh::{Mesh};
use crate::resource::Handle;
use crate::util;
use crate::Renderer;

use crate::backend::vk::command::{CommandBuffer, CommandError};
use crate::descriptor::DescriptorSet;
use crate::pipeline::{GraphicsPipeline, PipelineStorageReadGuard, ShaderStage};

use resurs::Async;

pub struct RenderPassBuilder<'a> {
    renderer: &'a Renderer,
    vertex_buffers: BufferStorageReadGuard<'a, Async<VertexBuffer>>,
    index_buffers: BufferStorageReadGuard<'a, Async<IndexBuffer>>,
    graphics_pipelines: PipelineStorageReadGuard<'a>,
    frame_idx: u32,
    command_buffer: CommandBuffer,
}

impl<'a> RenderPassBuilder<'a> {
    pub fn bind_shader_resource_group(
        &mut self,
        idx: u32,
        dset: &Handle<DescriptorSet>,
        pipeline: &Handle<GraphicsPipeline>,
    ) -> &mut Self {
        let dset = self
            .renderer
            .get_descriptor_set(dset)
            .expect("Failed to find descriptor set");

        let pipeline = self
            .graphics_pipelines
            .get(&pipeline.wrap_async())
            .expect("Failed to find pipeline")
            .as_ref()
            .expect("Should have arrived");

        self.command_buffer.bind_descriptor_set(idx, dset, pipeline);

        self
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &Handle<GraphicsPipeline>) -> &mut Self {
        let pipeline = self
            .graphics_pipelines
            .get(&pipeline.wrap_async())
            .expect("Failed to get pipeline")
            .as_ref()
            .expect("Should have arrived");

        self.command_buffer.bind_graphics_pipeline(pipeline);

        self
    }

    pub fn bind_index_buffer(&mut self, handle: &BufferHandle<IndexBuffer>) -> &mut Self {
        let ib = self
            .index_buffers
            .get(&handle.wrap_async(), self.frame_idx as usize)
            .expect("Failed to get index buffer")
            .as_ref()
            .expect("Should have arrived");

        self.command_buffer.bind_index_buffer(&ib, 0);

        self
    }

    pub fn bind_vertex_buffer(&mut self, handle: &BufferHandle<VertexBuffer>) -> &mut Self {
        let vb = self
            .vertex_buffers
            .get(&handle.wrap_async(), self.frame_idx as usize)
            .expect("Failed to get index buffer")
            .as_ref()
            .expect("Should have arrived");

        self.command_buffer.bind_vertex_buffer(&vb, 0);

        self
    }

    pub fn draw_mesh(&mut self, mesh: &Mesh) -> &mut Self {
        let vertex_index = mesh.vertex_buffer.idx() as i32;
        let indices_index = mesh.index_buffer.idx();
        let n_indices = mesh.index_buffer.n_elems();

        let vb = self
            .vertex_buffers
            .get(&mesh.vertex_buffer.wrap_async(), self.frame_idx as usize)
            .expect("Failed to get index buffer")
            .as_ref()
            .expect("Should have arrived");

        let ib = self
            .index_buffers
            .get(&mesh.index_buffer.wrap_async(), self.frame_idx as usize)
            .expect("Failed to get index buffer")
            .as_ref()
            .expect("Should have arrived");

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
            .graphics_pipelines
            .get(&pipeline.wrap_async())
            .expect("Failed to get pipeline");

        if let Async::Available(pipeline) = pipeline {
            self.command_buffer.bind_push_constant(pipeline, stage, v);
        }

        self
    }

    pub fn new(renderer: &'a Renderer, command_buffer: CommandBuffer, frame_idx: u32) -> Self {
        Self {
            renderer,
            vertex_buffers: renderer.async_resources.vertex_buffers.read(),
            index_buffers: renderer.async_resources.index_buffers.read(),
            graphics_pipelines: renderer.async_resources.graphics_pipelines.read(),
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
