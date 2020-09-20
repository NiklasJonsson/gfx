use ash::version::DeviceV1_0;
use ash::vk;

use thiserror::Error;

use crate::descriptor::DescriptorSet;
use crate::device::Device;
use crate::device::HasVkDevice;
use crate::device::VkDeviceHandle;
use crate::framebuffer::Framebuffer;
use crate::mesh::IndexBuffer;
use crate::mesh::VertexBuffer;
use crate::pipeline::GraphicsPipeline;
use crate::pipeline::Pipeline;
use crate::pipeline::ShaderStage;
use crate::queue::QueueFamily;
use crate::render_pass::RenderPass;
use crate::util;

// TODO: Temporary
use crate::mem::BufferHandle;
use crate::mesh::Mesh;
use crate::resource::Handle;
use crate::resource::ResourceManager;
use crate::Renderer;

#[derive(Debug, Error)]
pub enum CommandError {
    #[error("Command pool creation failed: {0}")]
    PoolCreation(vk::Result),
    #[error("Command pool reset failed: {0}")]
    PoolReset(vk::Result),
    #[error("Command buffer allocation failed: {0}")]
    BufferAlloc(vk::Result),
    #[error("Command buffer begin() failed: {0}")]
    BufferBegin(vk::Result),
    #[error("Command buffer end() failed: {0}")]
    BufferEnd(vk::Result),
}

pub struct CommandPool {
    queue_family: QueueFamily,
    vk_command_pool: vk::CommandPool,
    vk_device: VkDeviceHandle,
}

impl std::ops::Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.vk_device
                .destroy_command_pool(self.vk_command_pool, None);
        }
    }
}

impl CommandPool {
    fn new(device: &Device, qfam: QueueFamily) -> Result<Self, CommandError> {
        let info = vk::CommandPoolCreateInfo {
            queue_family_index: qfam.index,
            ..Default::default()
        };

        let vk_device = device.vk_device();

        let vk_command_pool = unsafe {
            vk_device
                .create_command_pool(&info, None)
                .map_err(CommandError::PoolCreation)?
        };

        Ok(Self {
            queue_family: qfam,
            vk_command_pool,
            vk_device,
        })
    }

    pub fn graphics(device: &Device) -> Result<Self, CommandError> {
        Self::new(device, device.graphics_queue_family().clone())
    }

    pub fn reset(&mut self) -> Result<(), CommandError> {
        unsafe {
            self.vk_device
                .reset_command_pool(self.vk_command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(CommandError::PoolReset)
        }
    }

    pub fn util(device: &Device) -> Result<Self, CommandError> {
        Self::new(device, device.util_queue_family().clone())
    }

    pub fn create_command_buffer(
        &self,
        submission_type: CommandBufferSubmission,
    ) -> Result<CommandBuffer, CommandError> {
        let mut r = self.create_command_buffers(1, submission_type)?;
        debug_assert_eq!(r.len(), 1);
        Ok(r.remove(0))
    }

    pub fn create_command_buffers(
        &self,
        amount: u32,
        submission_type: CommandBufferSubmission,
    ) -> Result<Vec<CommandBuffer>, CommandError> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.vk_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(amount);

        let allocated = unsafe {
            self.vk_device
                .allocate_command_buffers(&info)
                .map_err(CommandError::BufferAlloc)?
        };

        Ok(allocated
            .into_iter()
            .map(|vk_cmd_buf| {
                CommandBuffer::new(
                    VkDeviceHandle::clone(&self.vk_device),
                    vk_cmd_buf,
                    self.queue_family.props.queue_flags,
                    submission_type,
                )
            })
            .collect::<Result<Vec<CommandBuffer>, CommandError>>()?)
    }

    pub fn begin_single_submit(&self) -> Result<CommandBuffer, CommandError> {
        self.create_command_buffer(CommandBufferSubmission::Single)
    }
}

#[derive(Copy, Clone, Debug)]
pub enum CommandBufferSubmission {
    Single,
    Multi,
}

pub struct CommandBuffer {
    queue_flags: vk::QueueFlags,
    vk_cmd_buffer: vk::CommandBuffer,
    vk_device: VkDeviceHandle,
    is_started: bool,
}

impl CommandBuffer {
    fn new(
        vk_device: VkDeviceHandle,
        vk_cmd_buffer: vk::CommandBuffer,
        queue_flags: vk::QueueFlags,
        submission_type: CommandBufferSubmission,
    ) -> Result<Self, CommandError> {
        let flags = match submission_type {
            CommandBufferSubmission::Single => vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            _ => vk::CommandBufferUsageFlags::empty(),
        };

        let info = vk::CommandBufferBeginInfo {
            flags,
            ..Default::default()
        };

        unsafe {
            vk_device
                .begin_command_buffer(vk_cmd_buffer, &info)
                .map_err(CommandError::BufferBegin)?;
        };

        Ok(Self {
            vk_cmd_buffer,
            vk_device,
            queue_flags,
            is_started: true,
        })
    }

    pub fn vk_command_buffer(&self) -> &vk::CommandBuffer {
        &self.vk_cmd_buffer
    }

    pub fn is_started(&self) -> bool {
        self.is_started
    }

    pub fn end(&mut self) -> Result<&mut Self, CommandError> {
        unsafe {
            self.vk_device
                .end_command_buffer(self.vk_cmd_buffer)
                .map_err(CommandError::BufferEnd)?;
        }
        Ok(self)
    }

    pub fn begin_render_pass(
        &mut self,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        extent: util::Extent2D,
    ) -> &mut Self {
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(*render_pass.vk_render_pass())
            .framebuffer(*framebuffer.vk_framebuffer())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: extent.into(),
            })
            .clear_values(render_pass.vk_clear_values());

        unsafe {
            self.vk_device.cmd_begin_render_pass(
                self.vk_cmd_buffer,
                &info,
                vk::SubpassContents::INLINE,
            );
        }

        self
    }

    pub fn end_render_pass(&mut self) -> &mut Self {
        unsafe {
            self.vk_device.cmd_end_render_pass(self.vk_cmd_buffer);
        }

        self
    }

    pub fn bind_graphics_pipeline(&mut self, graphics_pipeline: &GraphicsPipeline) -> &mut Self {
        assert!(self.queue_flags.contains(vk::QueueFlags::GRAPHICS));

        unsafe {
            self.vk_device.cmd_bind_pipeline(
                self.vk_cmd_buffer,
                GraphicsPipeline::BIND_POINT,
                *graphics_pipeline.vk_pipeline(),
            );
        }

        self
    }

    pub fn bind_vertex_buffer(&mut self, buffer: &VertexBuffer, offset: u64) -> &mut Self {
        assert!(self.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        log::trace!("binding vertex buffer {:?} at {}", buffer, offset);

        unsafe {
            self.vk_device.cmd_bind_vertex_buffers(
                self.vk_cmd_buffer,
                0,
                &[*buffer.vk_buffer()],
                &[offset],
            );
        }

        self
    }

    pub fn set_scissor(&mut self, scissor: util::Rect2D) -> &mut Self {
        unsafe {
            self.vk_device
                .cmd_set_scissor(self.vk_cmd_buffer, 0, &[vk::Rect2D::from(scissor)]);
        }

        self
    }

    pub fn set_viewport(&mut self, viewport: util::Viewport) -> &mut Self {
        unsafe {
            self.vk_device
                .cmd_set_viewport(self.vk_cmd_buffer, 0, &[vk::Viewport::from(viewport)]);
        }

        self
    }

    pub fn bind_index_buffer(&mut self, buffer: &IndexBuffer, offset: u64) -> &mut Self {
        assert!(self.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        log::trace!("binding index buffer {:?} at {}", buffer, offset);

        unsafe {
            self.vk_device.cmd_bind_index_buffer(
                self.vk_cmd_buffer,
                *buffer.vk_buffer(),
                offset,
                buffer.vk_index_type(),
            );
        }

        self
    }

    pub fn bind_descriptor_set(
        &mut self,
        idx: u32,
        set: &DescriptorSet,
        pipeline: &GraphicsPipeline,
    ) -> &mut Self {
        assert!(self.queue_flags.contains(vk::QueueFlags::GRAPHICS));

        let sets = [*set.vk_descriptor_set()];
        unsafe {
            self.vk_device.cmd_bind_descriptor_sets(
                self.vk_cmd_buffer,
                GraphicsPipeline::BIND_POINT,
                *pipeline.vk_pipeline_layout(),
                idx,
                &sets,
                &[],
            );
        }

        self
    }

    pub fn draw_indexed(
        &mut self,
        n_vertices: u32,
        indices_index: u32,
        vertices_index: i32,
    ) -> &mut Self {
        assert!(self.queue_flags.contains(vk::QueueFlags::GRAPHICS));

        unsafe {
            self.vk_device.cmd_draw_indexed(
                self.vk_cmd_buffer,
                n_vertices,
                1,
                indices_index,
                vertices_index,
                0,
            );
        }

        self
    }

    pub fn copy_buffer(&mut self, src: &vk::Buffer, dst: &vk::Buffer, size: usize) -> &mut Self {
        let info = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: size as u64,
        };

        unsafe {
            self.vk_device
                .cmd_copy_buffer(self.vk_cmd_buffer, *src, *dst, &[info]);
        }

        self
    }

    pub fn copy_buffer_to_image(
        &mut self,
        src: &vk::Buffer,
        dst: &vk::Image,
        extent: &util::Extent2D,
    ) -> &mut Self {
        // TODO: Read this info from dst (by passing not just the vk::Image)
        let info = vk::BufferImageCopy {
            buffer_offset: 0,
            // For e.g. padded rows
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
        };

        unsafe {
            self.vk_device.cmd_copy_buffer_to_image(
                self.vk_cmd_buffer,
                *src,
                *dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[info],
            );
        }

        self
    }

    pub fn pipeline_barrier(
        &mut self,
        barrier: &vk::ImageMemoryBarrier,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) -> &mut Self {
        unsafe {
            self.vk_device.cmd_pipeline_barrier(
                self.vk_cmd_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[*barrier],
            );
        }

        self
    }

    pub fn blit_image(
        &mut self,
        src: &vk::Image,
        dst: &vk::Image,
        vk_image_blit: &vk::ImageBlit,
    ) -> &mut Self {
        unsafe {
            self.vk_device.cmd_blit_image(
                self.vk_cmd_buffer,
                *src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                *dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[*vk_image_blit],
                vk::Filter::LINEAR,
            );
        }

        self
    }

    pub fn bind_push_constant<V>(
        &mut self,
        pipeline: &GraphicsPipeline,
        stage: ShaderStage,
        v: &V,
    ) -> &mut Self {
        let bytes = util::as_bytes(v);
        assert!(bytes.len() <= 128);
        unsafe {
            self.vk_device.cmd_push_constants(
                self.vk_cmd_buffer,
                *pipeline.vk_pipeline_layout(),
                stage.into(),
                0,
                bytes,
            )
        }

        self
    }
}

// TODO: Rename RenderPassBuilder?
pub struct CommandBufferBuilder<'a> {
    renderer: &'a Renderer,
    command_buffer: CommandBuffer,
}

impl<'a> CommandBufferBuilder<'a> {
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
            .renderer
            .get_resource(pipeline)
            .expect("Failed to find pipeline");
        self.command_buffer.bind_descriptor_set(idx, dset, pipeline);

        self
    }

    pub fn bind_graphics_pipeline(&mut self, gfx_pipeline: &Handle<GraphicsPipeline>) -> &mut Self {
        let p = self
            .renderer
            .get_resource(gfx_pipeline)
            .expect("Failed to get pipeline");
        self.command_buffer.bind_graphics_pipeline(p);

        self
    }

    pub fn bind_index_buffer(&mut self, handle: &BufferHandle<IndexBuffer>) -> &mut Self {
        let ib = self
            .renderer
            .get_resource(handle)
            .expect("Failed to get index buffer");

        self.command_buffer.bind_index_buffer(ib, 0);

        self
    }

    pub fn bind_vertex_buffer(&mut self, handle: &BufferHandle<VertexBuffer>) -> &mut Self {
        let vb = self
            .renderer
            .get_resource(handle)
            .expect("Failed to get vertex buffer");

        self.command_buffer.bind_vertex_buffer(vb, 0);

        self
    }

    pub fn draw_mesh(&mut self, mesh: &Mesh) -> &mut Self {
        let vertex_index = mesh.vertex_buffer.idx() as i32;
        let indices_index = mesh.index_buffer.idx();
        let n_indices = mesh.index_buffer.n_elems();

        let ib = self
            .renderer
            .get_resource(&mesh.index_buffer)
            .expect("Failed to get index buffer");

        let vb = self
            .renderer
            .get_resource(&mesh.vertex_buffer)
            .expect("Failed to get vertex buffer");

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

    pub fn bind_push_constant<V>(
        &mut self,
        pipeline: &Handle<GraphicsPipeline>,
        stage: ShaderStage,
        v: &V,
    ) -> &mut Self {
        let p = self
            .renderer
            .get_resource(pipeline)
            .expect("Failed to get pipeline");
        self.command_buffer.bind_push_constant(p, stage, v);

        self
    }

    pub fn new(renderer: &'a Renderer, command_buffer: CommandBuffer) -> Self {
        Self {
            renderer,
            command_buffer,
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
