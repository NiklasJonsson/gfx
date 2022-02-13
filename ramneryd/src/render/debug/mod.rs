use super::pipeline;
use crate::math::Vec3;

use trekanten::buffer::{DeviceUniformBuffer, DeviceVertexBuffer};
use trekanten::descriptor::DescriptorSet;
use trekanten::pipeline::GraphicsPipeline;
use trekanten::resource::{MutResourceManager as _, ResourceManager as _};
use trekanten::vertex::{VertexDefinition, VertexFormat};
use trekanten::{BufferHandle, Handle};
use trekanten::{RenderPass, RenderPassEncoder};

use std::sync::Mutex;

pub mod bounding_box;
pub mod camera_frustum;
pub mod light;
pub mod window;

pub use camera_frustum::DrawFrustum;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct Vertex {
    pos: Vec3,
}

impl VertexDefinition for Vertex {
    fn format() -> VertexFormat {
        VertexFormat::builder()
            .add_attribute(trekanten::util::Format::FLOAT3)
            .build()
    }
}

pub enum DrawCmd {
    Lines { start: u32, count: u32 },
}

/// Queue simple one-off rendering commands
pub struct DebugRenderer {
    line_buffer: Vec<Vertex>,
    line_buffer_device: Option<BufferHandle<DeviceVertexBuffer>>,
    queue: Vec<DrawCmd>,
    pipeline: Handle<GraphicsPipeline>,
    view_data_srg: Handle<DescriptorSet>,
}

pub type DebugRendererRes = Mutex<DebugRenderer>;

impl DebugRenderer {
    pub fn draw_line_strip(&mut self, points: &[Vec3]) {
        assert!(!points.is_empty());
        let start: u32 = self.line_buffer.len().try_into().expect("Too big!");
        let count: u32 = points.len().try_into().expect("Too big!");
        self.line_buffer.reserve(points.len());
        self.line_buffer
            .extend(points.iter().map(|&pos| Vertex { pos }));
        self.queue.push(DrawCmd::Lines { start, count });
    }
}

/// Renderer interaction
impl DebugRenderer {
    pub fn new(
        shader_compiler: &pipeline::ShaderCompiler,
        render_pass: &Handle<RenderPass>,
        view_data_buf: &BufferHandle<DeviceUniformBuffer>,
        renderer: &mut trekanten::Renderer,
    ) -> Self {
        use super::uniform::UniformBlock as _;
        use trekanten::pipeline::{
            GraphicsPipelineDescriptor, PolygonMode, PrimitiveTopology, ShaderDescriptor,
            ShaderStage, TriangleCulling,
        };

        let shader_resource_group = DescriptorSet::builder(renderer)
            .add_buffer(
                view_data_buf,
                super::uniform::ViewData::BINDING,
                ShaderStage::VERTEX,
            )
            .build();

        let vertex_format = Vertex::format();

        let vertex = shader_compiler
            .compile(
                &pipeline::Defines::empty(),
                "world_pos_only_vert.glsl",
                pipeline::ShaderType::Vertex,
            )
            .expect("Failed to compile vert shader for debug renderer");

        let fragment = shader_compiler
            .compile(
                &pipeline::Defines::empty(),
                "red_frag.glsl",
                pipeline::ShaderType::Fragment,
            )
            .expect("Failed to compile frag shader for debug renderer");

        let pipeline_desc = GraphicsPipelineDescriptor::builder()
            .vert(ShaderDescriptor::FromRawSpirv(vertex.data()))
            .frag(ShaderDescriptor::FromRawSpirv(fragment.data()))
            .vertex_format(vertex_format)
            .culling(TriangleCulling::None)
            .polygon_mode(PolygonMode::Line)
            .primitive_topology(PrimitiveTopology::LineStrip)
            .build()
            .expect("Failed to build pipeline descriptor for debug renderer");

        let pipeline = renderer
            .create_gfx_pipeline(pipeline_desc, render_pass)
            .expect("Failed to create pipeline for shadow");

        Self {
            line_buffer: Vec::new(),
            line_buffer_device: None,
            queue: Vec::new(),
            pipeline,
            view_data_srg: shader_resource_group,
        }
    }

    pub fn upload<'a>(&mut self, frame: &mut trekanten::Frame<'a>) {
        use trekanten::buffer::{BufferMutability, VertexBufferDescriptor};

        if self.line_buffer.is_empty() {
            assert!(self.queue.is_empty());
            return;
        }
        assert!(!self.queue.is_empty());

        let desc = VertexBufferDescriptor::from_slice(&self.line_buffer, BufferMutability::Mutable);
        let vertex_buffer = if let Some(buf) = self.line_buffer_device {
            frame
                .recreate_resource_blocking(buf, desc)
                .expect("Bad vbuf handle")
        } else {
            frame
                .create_resource_blocking(desc)
                .expect("Failed to create vertex buffer for debug renderer")
        };

        self.line_buffer.clear();
        self.line_buffer_device = Some(vertex_buffer);
    }

    /// Note that the scene view data needs to have been bound to set 0, idx 0 already.
    pub fn record_commands(&mut self, renderpass: &mut RenderPassEncoder) {
        assert!(
            self.line_buffer.is_empty(),
            "Recorded more commands after upload?"
        );

        if self.queue.is_empty() {
            log::debug!("Draw queue is empty");
            return;
        }

        let vbuf = if let Some(vbuf) = self.line_buffer_device {
            vbuf
        } else {
            log::debug!("No vertex buffer to draw!");
            return;
        };

        log::trace!("Drawing {} line commands", self.queue.len());

        let range_all = 0..self.queue.len();
        renderpass.bind_graphics_pipeline(&self.pipeline);
        renderpass.bind_shader_resource_group(0, &self.view_data_srg, &self.pipeline);
        renderpass.bind_vertex_buffer(&vbuf);
        for cmd in self.queue.drain(range_all) {
            match cmd {
                DrawCmd::Lines { start, count } => {
                    renderpass.draw(count, start);
                }
            }
        }
    }
}

pub fn register_systems(builder: crate::ecs::ExecutorBuilder) -> crate::ecs::ExecutorBuilder {
    register_module_systems!(builder, window, camera_frustum, light, bounding_box)
}
