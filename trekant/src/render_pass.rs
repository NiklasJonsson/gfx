use crate::buffer::BufferHandle;
use crate::resource::Handle;
use crate::util;

use crate::backend;
use crate::pipeline::{GraphicsPipeline, ShaderStage};
use crate::pipeline_resource::PipelineResourceSet;
use crate::resource::Resources;
use crate::traits::PushConstant;
use crate::vk;

use backend::command::{CommandBuffer, CommandError};

pub struct RenderPassEncoder<'a> {
    resources: &'a Resources,
    frame_idx: u32,
    command_buffer: CommandBuffer,
}

impl<'a> RenderPassEncoder<'a> {
    pub fn bind_shader_resource_group(
        &mut self,
        idx: u32,
        dset: &Handle<PipelineResourceSet>,
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
            .get(pipeline)
            .expect("Failed to find pipeline");

        self.command_buffer.bind_descriptor_set(idx, dset, pipeline);

        self
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &Handle<GraphicsPipeline>) -> &mut Self {
        let pipeline = self
            .resources
            .graphics_pipelines
            .get(pipeline)
            .expect("Failed to get pipeline");

        self.command_buffer.bind_graphics_pipeline(pipeline);

        self
    }

    pub fn bind_index_buffer(&mut self, handle: BufferHandle) -> &mut Self {
        let ib = self
            .resources
            .buffers
            .get(handle, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        self.command_buffer.bind_index_buffer(ib, 0);

        self
    }

    pub fn bind_vertex_buffer(&mut self, handle: BufferHandle) -> &mut Self {
        let vb = self
            .resources
            .buffers
            .get(handle, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        self.command_buffer.bind_vertex_buffer(vb, 0);

        self
    }

    pub fn draw_mesh(
        &mut self,
        vertex_buffer: BufferHandle,
        index_buffer: BufferHandle,
    ) -> &mut Self {
        let vertex_index = vertex_buffer.offset() as i32;
        let indices_index = index_buffer.offset();
        let n_indices = index_buffer.len();

        let vb = self
            .resources
            .buffers
            .get(vertex_buffer, self.frame_idx as usize)
            .expect("Failed to get index buffer");

        let ib = self
            .resources
            .buffers
            .get(index_buffer, self.frame_idx as usize)
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

    pub fn draw(&mut self, n_vertices: u32, vertices_index: u32) -> &mut Self {
        self.command_buffer.draw(n_vertices, vertices_index);

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

    pub fn bind_push_constant<V: PushConstant>(
        &mut self,
        pipeline: &Handle<GraphicsPipeline>,
        stage: ShaderStage,
        v: &V,
    ) -> &mut Self {
        let pipeline = self
            .resources
            .graphics_pipelines
            .get(pipeline)
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

    pub fn end(mut self) -> Result<CommandBuffer, CommandError> {
        self.command_buffer.end_render_pass();
        Ok(self.command_buffer)
    }

    pub fn inner(self) -> CommandBuffer {
        let Self { command_buffer, .. } = self;
        command_buffer
    }
}

use crate::backend::render_pass::RenderPass as BackendRenderPass;

pub struct RenderPass(pub(crate) BackendRenderPass);

impl RenderPass {
    pub fn presentation_render_pass(
        device: &backend::device::Device,
        format: util::Format,
        msaa_sample_count: u8,
    ) -> Result<Self, crate::error::RenderError> {
        let msaa_sample_count = backend::n_to_sample_count(msaa_sample_count);
        let msaa_color_attach = vk::AttachmentDescription::builder()
            .format(vk::Format::from(format))
            .samples(msaa_sample_count)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let resolve_color_attach = vk::AttachmentDescription::builder()
            .format(vk::Format::from(format))
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let msaa_color_attach_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let resolve_color_attach_ref = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attach = vk::AttachmentDescription::builder()
            .format(device.depth_buffer_format())
            .samples(msaa_sample_count)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attach_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attach_refs = [msaa_color_attach_ref];
        let resolve_attach_refs = [resolve_color_attach_ref];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attach_refs)
            .resolve_attachments(&resolve_attach_refs)
            .depth_stencil_attachment(&depth_attach_ref);

        let attachments = [*msaa_color_attach, *depth_attach, *resolve_color_attach];
        let subpasses = [*subpass];

        let subpass_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let dependencies = [subpass_dependency.build()];

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        Self::new_vk(device, &create_info)
    }
}

impl RenderPass {
    pub fn new_vk(
        device: &backend::device::Device,
        create_info: &vk::RenderPassCreateInfo,
    ) -> Result<Self, crate::error::RenderError> {
        let rp = BackendRenderPass::new(device, create_info)
            .map_err(crate::error::RenderError::RenderPass)?;
        Ok(Self(rp))
    }
}
