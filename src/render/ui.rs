use trekanten::command::CommandBufferBuilder;
use trekanten::descriptor::DescriptorSet;
use trekanten::mesh::{BufferMutability, IndexBuffer, IndexBufferDescriptor};
use trekanten::mesh::{VertexBuffer, VertexBufferDescriptor};
use trekanten::pipeline::{
    BlendState, DepthTest, GraphicsPipeline, GraphicsPipelineDescriptor, ShaderDescriptor,
    ShaderStage, TriangleCulling,
};
use trekanten::resource::{MutResourceManager, ResourceManager};
use trekanten::texture::{MipMaps, Texture, TextureDescriptor};
use trekanten::util::{Extent2D, Format, Offset2D, Rect2D, Viewport};
use trekanten::vertex::{VertexDefinition, VertexFormat};
use trekanten::Frame;
use trekanten::Renderer;
use trekanten::{BufferHandle, Handle};

use specs::world::WorldExt;
use specs::World;

use imgui::im_str;

struct ImGuiVertex {
    _pos: [f32; 2],
    _uv: [f32; 2],
    _col: [u8; 4],
}

impl VertexDefinition for ImGuiVertex {
    fn format() -> VertexFormat {
        VertexFormat::builder()
            .add_attribute(Format::FLOAT2)
            .add_attribute(Format::FLOAT2)
            .add_attribute(Format::RGBA_UNORM)
            .build()
    }
}

static RAW_VERT_SPV: &[u32] = inline_spirv::include_spirv!(
    "src/render/shaders/imgui/vert.glsl",
    vert,
    glsl,
    entry = "main"
);
static RAW_FRAG_SPV: &[u32] = inline_spirv::include_spirv!(
    "src/render/shaders/imgui/frag.glsl",
    frag,
    glsl,
    entry = "main"
);

#[derive(Clone, Copy, Debug)]
struct PerFrameData {
    vertex_buffer: BufferHandle<VertexBuffer>,
    index_buffer: BufferHandle<IndexBuffer>,
    fb_width: f32,
    fb_height: f32,
}

// TODO: Move this to app?
pub struct UIContext {
    imgui: imgui::Context,
    _font_texture: Handle<Texture>,
    pipeline: Handle<GraphicsPipeline>,
    desc_set: Handle<DescriptorSet>,
    per_frame_data: Option<PerFrameData>,
}

fn build_ui<'a>(world: &World, ui: &imgui::Ui<'a>, pos: [f32; 2]) -> [f32; 2] {
    let dt = world.read_resource::<crate::time::DeltaTime>();

    let size = [300.0, 65.0];
    imgui::Window::new(im_str!("Global stats"))
        .size(size, imgui::Condition::FirstUseEver)
        .position(pos, imgui::Condition::FirstUseEver)
        .build(&ui, || {
            ui.text(im_str!("FPS: {:.3}", dt.as_fps()));
            ui.text(im_str!(
                "Cam pos: {}",
                super::ActiveCamera::camera_pos(world)
            ));
        });
    size
}

impl UIContext {
    pub fn new(renderer: &mut Renderer) -> Self {
        log::trace!("Setup ui resources");

        let mut ctx = imgui::Context::create();
        ctx.set_renderer_name(Some(imgui::ImString::from(format!(
            "ramneryd {}",
            env!("CARGO_PKG_VERSION")
        ))));
        ctx.io_mut()
            .backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);
        let extent = renderer.swapchain_extent();
        ctx.io_mut().display_size = [extent.width as f32, extent.height as f32];

        let font_texture = {
            let mut fonts = ctx.fonts();
            let atlas_texture = fonts.build_rgba32_texture();

            let tex_desc = TextureDescriptor::raw(
                atlas_texture.data,
                Extent2D {
                    width: atlas_texture.width,
                    height: atlas_texture.height,
                },
                Format::RGBA_UNORM,
                MipMaps::None,
            );
            renderer
                .create_resource(tex_desc)
                .expect("Failed to create font texture")
        };

        let vert = ShaderDescriptor::FromRawSpirv(RAW_VERT_SPV.to_vec());
        let frag = ShaderDescriptor::FromRawSpirv(RAW_FRAG_SPV.to_vec());
        let pipeline_descriptor = GraphicsPipelineDescriptor::builder()
            .vert(vert)
            .frag(frag)
            .vertex_format(ImGuiVertex::format())
            .culling(TriangleCulling::None)
            .blend_state(BlendState::Enabled)
            .depth_testing(DepthTest::Disabled)
            .build()
            .expect("Failed to builder graphics pipeline descriptor");

        let pipeline = renderer
            .create_resource(pipeline_descriptor)
            .expect("Failed to create graphics pipeline");

        let desc_set = DescriptorSet::builder(renderer)
            .add_texture(&font_texture, 0, ShaderStage::Fragment)
            .build();

        log::trace!("Done");
        UIContext {
            imgui: ctx,
            pipeline,
            desc_set,
            _font_texture: font_texture,
            per_frame_data: None,
        }
    }

    pub fn build_ui<'a>(&mut self, world: &World, frame: &mut Frame<'a>) -> Option<UIDrawCommands> {
        log::trace!("Building ui");
        let ui = self.imgui.frame();

        let mut y_offset = 0.0;
        let funcs = [
            build_ui,
            crate::settings::build_ui,
            crate::game_state::build_ui,
            crate::io::input::build_ui,
        ];
        for func in funcs.iter() {
            let size = func(world, &ui, [0.0, y_offset]);
            y_offset += size[1];
        }

        let draw_data = ui.render();
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];

        if fb_width <= 0.0 || fb_height <= 0.0 {
            return None;
        }

        let scale = [
            2.0 / draw_data.display_size[0],
            2.0 / draw_data.display_size[1],
        ];

        let vertex_shader_data = VertexShaderData {
            scale_translate: [
                scale[0],
                scale[1],
                -1.0 - draw_data.display_pos[0] * scale[0],
                -1.0 - draw_data.display_pos[1] * scale[1],
            ],
        };

        if draw_data.total_idx_count == 0 || draw_data.total_vtx_count == 0 {
            log::trace!("No vertices or indices to render");
            return None;
        }

        // TODO: This is extra copies + allocation, avoid this by exposing nicer creation tools from VertexBufferDescriptor
        let mut vertices = Vec::with_capacity(draw_data.total_vtx_count as usize);
        let mut indices = Vec::with_capacity(draw_data.total_idx_count as usize);

        let mut commands = Vec::new();

        // Will project scissor/clipping rectangles into framebuffer space
        let clip_offset = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;

        // Render command lists
        // (Because we merged all buffers into a single one, we maintain our own offset into them)
        let mut global_vertices_idx = 0;
        let mut global_indices_idx = 0;
        for draw_list in draw_data.draw_lists() {
            vertices.extend_from_slice(trekanten::util::as_byte_slice(draw_list.vtx_buffer()));
            indices.extend_from_slice(draw_list.idx_buffer());
            for cmd in draw_list.commands() {
                use imgui::DrawCmd;
                use imgui::DrawCmdParams;
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                            DrawCmdParams {
                                clip_rect,
                                vtx_offset,
                                idx_offset,
                                ..
                            },
                    } => {
                        // Project scissor/clipping rectangles into framebuffer space
                        let mut clip_rect = [
                            (clip_rect[0] - clip_offset[0]) * clip_scale[0],
                            (clip_rect[1] - clip_offset[1]) * clip_scale[1],
                            (clip_rect[2] - clip_offset[0]) * clip_scale[0],
                            (clip_rect[3] - clip_offset[1]) * clip_scale[1],
                        ];

                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0
                        {
                            // Negative offsets are illegal for vkCmdSetScissor
                            if clip_rect[0] < 0.0 {
                                clip_rect[0] = 0.0;
                            }

                            if clip_rect[1] < 0.0 {
                                clip_rect[1] = 0.0;
                            }

                            let scissor = Rect2D {
                                offset: Offset2D {
                                    x: clip_rect[0] as i32,
                                    y: clip_rect[1] as i32,
                                },
                                extent: Extent2D {
                                    width: (clip_rect[2] - clip_rect[0]) as u32,
                                    height: (clip_rect[3] - clip_rect[1]) as u32,
                                },
                            };

                            commands.push(UIDrawCommand {
                                scissor,
                                vertices_idx: (vtx_offset + global_vertices_idx) as i32,
                                indices_idx: (idx_offset + global_indices_idx) as u32,
                                count: count as u32,
                            });
                        }
                    }
                    DrawCmd::ResetRenderState => {
                        unimplemented!("Dear ImGui ResetRenderState is not supported!")
                    }
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        use imgui::internal::RawWrapper;
                        callback(draw_list.raw(), raw_cmd)
                    },
                }
            }

            global_vertices_idx += draw_list.vtx_buffer().len();
            global_indices_idx += draw_list.idx_buffer().len();
        }

        assert_eq!(
            std::mem::size_of::<imgui::DrawVert>(),
            std::mem::size_of::<ImGuiVertex>(),
            "Mismatch in imgui vertex type"
        );
        let vbuf_desc = VertexBufferDescriptor::from_raw(
            &vertices,
            ImGuiVertex::format(),
            BufferMutability::Mutable,
        );
        let ibuf_desc = IndexBufferDescriptor::from_slice(&indices, BufferMutability::Mutable);

        let (vertex_buffer, index_buffer) = if world.has_value::<UIDrawCommands>() {
            let per_frame_data = world.read_resource::<UIDrawCommands>().per_frame_data;
            let vertex_buffer = frame
                .recreate_resource(per_frame_data.vertex_buffer, vbuf_desc)
                .expect("Bad vbuf handle");
            let index_buffer = frame
                .recreate_resource(per_frame_data.index_buffer, ibuf_desc)
                .expect("Bad ibuf handle");
            (vertex_buffer, index_buffer)
        } else {
            let vh = frame
                .renderer()
                .create_resource(vbuf_desc)
                .expect("Failed to create vertex buffer");
            let ih = frame
                .renderer()
                .create_resource(ibuf_desc)
                .expect("Failed to create index buffer");

            (vh, ih)
        };

        let per_frame_data = PerFrameData {
            vertex_buffer,
            index_buffer,
            fb_width,
            fb_height,
        };

        Some(UIDrawCommands {
            per_frame_data,
            pipeline: self.pipeline,
            desc_set: self.desc_set,
            vertex_shader_data,
            commands,
        })
    }
}

#[derive(Debug)]
struct UIDrawCommand {
    scissor: Rect2D,
    vertices_idx: i32,
    indices_idx: u32,
    count: u32,
}

#[derive(Debug)]
pub struct UIDrawCommands {
    per_frame_data: PerFrameData,
    pipeline: Handle<GraphicsPipeline>,
    desc_set: Handle<DescriptorSet>,
    vertex_shader_data: VertexShaderData,
    commands: Vec<UIDrawCommand>,
}

#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct VertexShaderData {
    scale_translate: [f32; 4],
}

impl UIDrawCommands {
    pub fn record_draw_commands<'b>(self, cmd_buf: &mut CommandBufferBuilder<'b>) {
        log::trace!("Recording draw commands!");
        let Self {
            per_frame_data:
                PerFrameData {
                    vertex_buffer,
                    index_buffer,
                    fb_width,
                    fb_height,
                },
            pipeline,
            desc_set,
            vertex_shader_data,
            commands,
        } = self;

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: fb_width,
            height: fb_height,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        cmd_buf
            .set_viewport(viewport)
            .bind_graphics_pipeline(&pipeline)
            .bind_index_buffer(&index_buffer)
            .bind_vertex_buffer(&vertex_buffer)
            .bind_shader_resource_group(0, &desc_set, &pipeline)
            .bind_push_constant(&pipeline, ShaderStage::Vertex, &vertex_shader_data);

        for cmd in commands.iter() {
            cmd_buf.set_scissor(cmd.scissor).draw_indexed(
                cmd.count,
                cmd.indices_idx,
                cmd.vertices_idx,
            );
        }
    }
}
