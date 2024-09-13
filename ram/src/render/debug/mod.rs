use super::imgui::UIModules;
use super::{geometry, shader};
use crate::math::{Rgba, Vec3};

use trekant::pipeline::GraphicsPipeline;
use trekant::pipeline_resource::PipelineResourceSet;
use trekant::{BufferDescriptor, BufferHandle, BufferMutability, Handle, PushConstant};
use trekant::{RenderPass, RenderPassEncoder};
use trekant::{VertexDefinition, VertexFormat};

use crate::render::shader::{Defines, PipelineService, PipelineSettings, Shader, Shaders};

use std::sync::Mutex;

pub mod bounding_box;
pub mod camera;
pub mod imdbg;
pub mod light;
pub mod window;

pub use camera::DrawFrustum;
pub use window::{OneShotDebugUI, OneShotDebugUIFunction};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
struct Vertex {
    pos: [f32; 3],
}

unsafe impl VertexDefinition for Vertex {
    fn format() -> VertexFormat {
        VertexFormat::from(trekant::util::Format::FLOAT3)
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Color {
    v: [f32; 4],
}

unsafe impl PushConstant for Color {
    fn size() -> u16 {
        std::mem::size_of::<Color>() as u16
    }
}

pub struct LineConfig {
    pub color: Rgba,
}

enum DrawCmd {
    Lines {
        start: u32,
        count: u32,
        color: Color,
    },
}

/// Queue simple one-off rendering commands
struct DebugRendererDrawBuffer {
    line_buffer: Vec<Vertex>,
    line_buffer_gpu: Option<BufferHandle>,
    queue: Vec<DrawCmd>,
    pipeline: Handle<GraphicsPipeline>,
    view_data_srg: Handle<PipelineResourceSet>,
}

impl DebugRendererDrawBuffer {
    pub fn draw_line_strip(&mut self, points: &[Vec3], cfg: LineConfig) {
        assert!(!points.is_empty());
        let count: u32 = points.len().try_into().expect("Too big!");

        let start: u32 = self.line_buffer.len().try_into().expect("Too big!");
        self.line_buffer.reserve(points.len());
        self.line_buffer.extend(points.iter().map(|&pos| Vertex {
            pos: pos.into_array(),
        }));
        self.queue.push(DrawCmd::Lines {
            start,
            count,
            color: Color {
                v: cfg.color.into_array(),
            },
        });
    }
}

impl Default for LineConfig {
    fn default() -> Self {
        Self { color: Rgba::red() }
    }
}

pub struct DebugRenderer {
    draw_buffer: Mutex<DebugRendererDrawBuffer>,
}

impl DebugRenderer {
    pub fn draw_line_strip(&self, points: &[Vec3], cfg: LineConfig) {
        let mut db = self.draw_buffer.lock().unwrap();
        db.draw_line_strip(points, cfg);
    }

    pub fn draw_obb(&self, obb: crate::math::Obb, cfg: LineConfig) {
        let lines = geometry::obb_line_strip(obb);
        {
            let mut db = self.draw_buffer.lock().unwrap();
            db.draw_line_strip(&lines, cfg);
        }
    }

    pub fn draw_aabb(&self, aabb: crate::math::Aabb, cfg: LineConfig) {
        let lines = geometry::aabb_line_strip(aabb);
        {
            let mut db = self.draw_buffer.lock().unwrap();
            db.draw_line_strip(&lines, cfg);
        }
    }
}

/// Renderer interaction
impl DebugRenderer {
    pub fn new(
        pipeline_service: &PipelineService,
        render_pass: Handle<RenderPass>,
        view_data_buf: BufferHandle,
        renderer: &mut trekant::Renderer,
    ) -> Self {
        use trekant::pipeline::{PolygonMode, PrimitiveTopology, ShaderStage, TriangleCulling};

        let shader_resource_group = PipelineResourceSet::builder(renderer)
            .add_buffer(view_data_buf, 0, ShaderStage::VERTEX)
            .build();

        let vertex_format = Vertex::format();

        let vert = Shader {
            loc: super::shader_path("world_pos_only_vert.glsl"),
            defines: Defines::empty(),
            debug_name: Some("debug-lines-vert".to_owned()),
        };

        let frag = Shader {
            loc: super::shader_path("push_constant_color_frag.glsl"),
            defines: shader::Defines::empty(),
            debug_name: Some("debug-lines-frag".to_owned()),
        };

        let shaders = Shaders {
            vert,
            frag: Some(frag),
        };

        let settings = PipelineSettings {
            vertex_format,
            culling: TriangleCulling::None,
            polygon_mode: PolygonMode::Line,
            primitive_topology: PrimitiveTopology::LineStrip,
            ..Default::default()
        };

        let pipeline = pipeline_service
            .create(shaders, settings, render_pass, renderer)
            .expect("Failed to create pipeline for debug renderer");

        Self {
            draw_buffer: Mutex::new(DebugRendererDrawBuffer {
                line_buffer: Vec::new(),
                line_buffer_gpu: None,
                queue: Vec::new(),
                pipeline,
                view_data_srg: shader_resource_group,
            }),
        }
    }

    pub fn upload(&self, frame: &mut trekant::Frame<'_>) {
        let mut db = self.draw_buffer.lock().unwrap();

        if db.line_buffer.is_empty() {
            assert!(db.queue.is_empty());
            return;
        }
        assert!(!db.queue.is_empty());

        let desc =
            BufferDescriptor::vertex_buffer(&db.line_buffer as &[_], BufferMutability::Mutable);
        let vertex_buffer = frame
            .create_buffer_ext(desc, db.line_buffer_gpu)
            .expect("Failed to create vertex buffer for debug renderer");

        db.line_buffer.clear();
        db.line_buffer_gpu = Some(vertex_buffer);
    }

    /// Note that the scene view data needs to have been bound to set 0, idx 0 already.
    pub fn record_commands(&self, renderpass: &mut RenderPassEncoder) {
        let mut db = self.draw_buffer.lock().unwrap();

        assert!(
            db.line_buffer.is_empty(),
            "Recorded more commands after upload?"
        );

        if db.queue.is_empty() {
            log::debug!("Draw queue is empty");
            return;
        }

        let vbuf = if let Some(vbuf) = db.line_buffer_gpu {
            vbuf
        } else {
            log::debug!("No vertex buffer for debug renderer to draw!");
            return;
        };

        log::trace!("Drawing {} line commands", db.queue.len());

        let pipeline = db.pipeline;
        renderpass.bind_graphics_pipeline(&pipeline);
        renderpass.bind_shader_resource_group(0, &db.view_data_srg, &db.pipeline);
        renderpass.bind_vertex_buffer(vbuf);
        for cmd in db.queue.drain(..) {
            match cmd {
                DrawCmd::Lines {
                    start,
                    count,
                    color,
                } => {
                    renderpass.bind_push_constant(
                        &pipeline,
                        trekant::ShaderStage::FRAGMENT,
                        &color,
                    );
                    renderpass.draw(count, start);
                }
            }
        }
    }
}

pub fn register_systems(builder: crate::ecs::ExecutorBuilder) -> crate::ecs::ExecutorBuilder {
    register_module_systems!(builder, window, camera, light, bounding_box)
}

pub fn register_ui_modules(mods: &mut UIModules) {
    mods.push(window::ui_module());
    mods.push(imdbg::ui_module());
}
