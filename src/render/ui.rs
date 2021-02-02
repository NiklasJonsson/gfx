use trekanten::descriptor::DescriptorSet;
use trekanten::mem::{
    BufferMutability, IndexBuffer, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor,
    VertexBuffer,
};
use trekanten::pipeline::{
    BlendState, DepthTest, GraphicsPipeline, GraphicsPipelineDescriptor, ShaderDescriptor,
    ShaderStage, TriangleCulling,
};
use trekanten::resource::{MutResourceManager, ResourceManager};
use trekanten::texture::{MipMaps, Texture, TextureDescriptor};
use trekanten::util::{Extent2D, Format, Offset2D, Rect2D, Viewport};
use trekanten::vertex::{VertexDefinition, VertexFormat};
use trekanten::Frame;
use trekanten::RenderPassEncoder;
use trekanten::Renderer;
use trekanten::{BufferHandle, Handle};

use crate::common::Name;
use crate::io::input;
use crate::io::input::KeyCode;
use crate::render::pipeline::{Defines, ShaderCompiler, ShaderType};
use crate::time::DeltaTime;

use specs::world::WorldExt;
use specs::World;

use imgui::im_str;

use std::path::Path;

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

#[derive(Clone, Copy, Debug)]
struct PerFrameData {
    vertex_buffer: BufferHandle<VertexBuffer>,
    index_buffer: BufferHandle<IndexBuffer>,
    fb_width: f32,
    fb_height: f32,
}

pub struct UIContext {
    imgui: imgui::Context,
    _font_texture: Handle<Texture>,
    pipeline: Handle<GraphicsPipeline>,
    desc_set: Handle<DescriptorSet>,
    input_entity: specs::Entity,
    per_frame_data: Option<PerFrameData>,
}

/* TODO:
    /// Render preparation callback.
    ///
    /// Call this before calling the imgui-rs UI `render_with`/`render` function.
    /// This function performs the following actions:
    ///
    /// * mouse cursor is changed and/or hidden (if requested by imgui-rs)
    pub fn prepare_render(&mut self, ui: &Ui, window: &Window) {
        let io = ui.io();
        if !io
            .config_flags
            .contains(ConfigFlags::NO_MOUSE_CURSOR_CHANGE)
        {
            let cursor = CursorSettings {
                cursor: ui.mouse_cursor(),
                draw_cursor: io.mouse_draw_cursor,
            };
            if self.cursor_cache != Some(cursor) {
                cursor.apply(window);
                self.cursor_cache = Some(cursor);
            }
        }
    }
}
*/

const MOUSE_WHEEL_DELTA_X: input::RangeId = input::RangeId(0);
const MOUSE_WHEEL_DELTA_Y: input::RangeId = input::RangeId(1);

const MOUSE_BUTTON_SEPARATOR: u32 = 1 << 16;
fn is_mouse_button(state_id: input::StateId) -> bool {
    state_id.0 >= MOUSE_BUTTON_SEPARATOR && (state_id.0 < (MOUSE_BUTTON_SEPARATOR + 5))
}

const fn is_keyboard_button(key: u32) -> bool {
    key < MOUSE_BUTTON_SEPARATOR
}

fn mouse_button_stateid(btn: input::MouseButton) -> input::StateId {
    let i = match btn {
        input::MouseButton::Left => 0,
        input::MouseButton::Right => 1,
        input::MouseButton::Middle => 2,
        input::MouseButton::Other(idx) => 3 + idx,
    };

    input::StateId(i as u32 + MOUSE_BUTTON_SEPARATOR)
}

const fn mouse_stateid_idx(s: input::StateId) -> u32 {
    s.0 - MOUSE_BUTTON_SEPARATOR
}

const NAME: &str = "UIInputContext";

impl UIContext {
    fn init_imgui_ctx() -> imgui::Context {
        let mut ctx = imgui::Context::create();

        ctx.set_renderer_name(Some(im_str!(
            "ramneryd-trekanten {}",
            env!("CARGO_PKG_VERSION")
        )));
        ctx.set_platform_name(Some(im_str!(
            "ramneryd-winit {}",
            env!("CARGO_PKG_VERSION")
        )));

        let io = ctx.io_mut();

        io.backend_flags
            .insert(imgui::BackendFlags::RENDERER_HAS_VTX_OFFSET);
        io.backend_flags
            .insert(imgui::BackendFlags::HAS_MOUSE_CURSORS);
        io[imgui::Key::Tab] = KeyCode::Tab as _;
        io[imgui::Key::LeftArrow] = KeyCode::Left as _;
        io[imgui::Key::RightArrow] = KeyCode::Right as _;
        io[imgui::Key::UpArrow] = KeyCode::Up as _;
        io[imgui::Key::DownArrow] = KeyCode::Down as _;
        io[imgui::Key::PageUp] = KeyCode::PageUp as _;
        io[imgui::Key::PageDown] = KeyCode::PageDown as _;
        io[imgui::Key::Home] = KeyCode::Home as _;
        io[imgui::Key::End] = KeyCode::End as _;
        io[imgui::Key::Insert] = KeyCode::Insert as _;
        io[imgui::Key::Delete] = KeyCode::Delete as _;
        io[imgui::Key::Backspace] = KeyCode::Back as _;
        io[imgui::Key::Space] = KeyCode::Space as _;
        io[imgui::Key::Enter] = KeyCode::Return as _;
        io[imgui::Key::Escape] = KeyCode::Escape as _;
        io[imgui::Key::KeyPadEnter] = KeyCode::NumpadEnter as _;
        io[imgui::Key::A] = KeyCode::A as _;
        io[imgui::Key::C] = KeyCode::C as _;
        io[imgui::Key::V] = KeyCode::V as _;
        io[imgui::Key::X] = KeyCode::X as _;
        io[imgui::Key::Y] = KeyCode::Y as _;
        io[imgui::Key::Z] = KeyCode::Z as _;

        ctx
    }

    fn resize(&mut self, extent: Extent2D) {
        self.imgui.io_mut().display_size = [extent.width as f32, extent.height as f32];
    }

    fn create_input_context(
        wants_mouse: bool,
        wants_keyboard: bool,
        wants_text: bool,
    ) -> Result<input::InputContext, input::InputContextError> {
        use input::MouseButton;
        use input::{DeviceAxis, InputContextPriority, StateId};
        let mouse = if wants_mouse {
            input::InputPassthrough::Consume
        } else {
            input::InputPassthrough::Passthrough
        };
        let keyboard = if wants_keyboard {
            input::InputPassthrough::Consume
        } else {
            input::InputPassthrough::Passthrough
        };
        let text = if wants_text {
            input::InputPassthrough::Consume
        } else {
            input::InputPassthrough::Passthrough
        };

        log::trace!(
            "Creating input context for ui. mouse: {}, keyboard: {}, text: {}",
            wants_mouse,
            wants_keyboard,
            wants_text
        );

        let b = input::InputContext::builder("UIInputContext")
            .description("Input mapping for the dear imgui ui context")
            .priority(InputContextPriority::Ui)
            .with_state_passthrough(KeyCode::Tab, StateId(KeyCode::Tab as _), keyboard)?
            .with_state_passthrough(KeyCode::Left, StateId(KeyCode::Left as _), keyboard)?
            .with_state_passthrough(KeyCode::Right, StateId(KeyCode::Right as _), keyboard)?
            .with_state_passthrough(KeyCode::Up, StateId(KeyCode::Up as _), keyboard)?
            .with_state_passthrough(KeyCode::Down, StateId(KeyCode::Down as _), keyboard)?
            .with_state_passthrough(KeyCode::PageUp, StateId(KeyCode::PageUp as _), keyboard)?
            .with_state_passthrough(KeyCode::PageDown, StateId(KeyCode::PageDown as _), keyboard)?
            .with_state_passthrough(KeyCode::Home, StateId(KeyCode::Home as _), keyboard)?
            .with_state_passthrough(KeyCode::End, StateId(KeyCode::End as _), keyboard)?
            .with_state_passthrough(KeyCode::Insert, StateId(KeyCode::Insert as _), keyboard)?
            .with_state_passthrough(KeyCode::Delete, StateId(KeyCode::Delete as _), keyboard)?
            .with_state_passthrough(KeyCode::Back, StateId(KeyCode::Back as _), keyboard)?
            .with_state_passthrough(KeyCode::Space, StateId(KeyCode::Space as _), keyboard)?
            .with_state_passthrough(KeyCode::Return, StateId(KeyCode::Return as _), keyboard)?
            .with_state_passthrough(KeyCode::Escape, StateId(KeyCode::Escape as _), keyboard)?
            .with_state_passthrough(
                KeyCode::NumpadEnter,
                StateId(KeyCode::NumpadEnter as _),
                keyboard,
            )?
            .with_state_passthrough(KeyCode::A, StateId(KeyCode::A as _), keyboard)?
            .with_state_passthrough(KeyCode::C, StateId(KeyCode::C as _), keyboard)?
            .with_state_passthrough(KeyCode::V, StateId(KeyCode::V as _), keyboard)?
            .with_state_passthrough(KeyCode::X, StateId(KeyCode::X as _), keyboard)?
            .with_state_passthrough(KeyCode::Y, StateId(KeyCode::Y as _), keyboard)?
            .with_state_passthrough(KeyCode::Z, StateId(KeyCode::Z as _), keyboard)?
            .wants_cursor_pos(true, mouse)
            .with_range_passthrough(DeviceAxis::ScrollX, MOUSE_WHEEL_DELTA_X, 1.0, mouse)?
            .with_range_passthrough(DeviceAxis::ScrollY, MOUSE_WHEEL_DELTA_Y, 1.0, mouse)?
            .with_state_passthrough(
                MouseButton::Left,
                mouse_button_stateid(MouseButton::Left),
                mouse,
            )?
            .with_state_passthrough(
                MouseButton::Right,
                mouse_button_stateid(MouseButton::Right),
                mouse,
            )?
            .with_state_passthrough(
                MouseButton::Middle,
                mouse_button_stateid(MouseButton::Middle),
                mouse,
            )?
            .with_state_passthrough(
                MouseButton::Other(0),
                mouse_button_stateid(MouseButton::Other(0)),
                mouse,
            )?
            .with_state_passthrough(
                MouseButton::Other(1),
                mouse_button_stateid(MouseButton::Other(1)),
                mouse,
            )?
            .wants_text(true, text)
            .build();

        Ok(b)
    }

    fn init_entity(world: &mut World) -> specs::Entity {
        use specs::world::Builder as _;

        let input_ctx = Self::create_input_context(false, false, false)
            .expect("Failed to create inputo context for ui");

        world
            .create_entity()
            .with(input_ctx)
            .with(Name::from(NAME))
            .build()
    }

    pub fn new(renderer: &mut Renderer, world: &mut World) -> Self {
        log::trace!("Setup ui resources");

        let mut imgui_ctx = Self::init_imgui_ctx();

        let font_texture = {
            let mut fonts = imgui_ctx.fonts();
            let atlas_texture = fonts.build_rgba32_texture();

            // We get borrowed data from imgui for the texture so we need a copy here
            // TODO: This is a use case for supporting borrowed data in a synchronous api
            let tex_desc = TextureDescriptor::from_vec(
                atlas_texture.data.to_owned(),
                Extent2D {
                    width: atlas_texture.width,
                    height: atlas_texture.height,
                },
                Format::RGBA_UNORM,
                MipMaps::None,
            );
            renderer
                .create_resource_blocking(tex_desc)
                .expect("Failed to create font texture")
        };

        let (vert, frag) = {
            let compiler = world.read_resource::<ShaderCompiler>();
            let defines = Defines::default();
            let vert = compiler
                .compile(&defines, Path::new("imgui/vert.glsl"), ShaderType::Vertex)
                .expect("Failed to compile imgui vert");
            let frag = compiler
                .compile(&defines, Path::new("imgui/frag.glsl"), ShaderType::Fragment)
                .expect("Failed to compile imgui frag");
            (vert, frag)
        };

        let vert = ShaderDescriptor::FromRawSpirv(vert.data());
        let frag = ShaderDescriptor::FromRawSpirv(frag.data());
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
            .create_resource_blocking(pipeline_descriptor)
            .expect("Failed to create graphics pipeline");

        let desc_set = DescriptorSet::builder(renderer)
            .add_texture(&font_texture, 0, ShaderStage::FRAGMENT)
            .build();

        let input_entity = Self::init_entity(world);

        let mut ui_ctx = UIContext {
            imgui: imgui_ctx,
            pipeline,
            desc_set,
            _font_texture: font_texture,
            input_entity,
            per_frame_data: None,
        };

        ui_ctx.resize(renderer.swapchain_extent());

        log::trace!("Done");

        ui_ctx
    }

    pub fn pre_frame(&mut self, world: &World) {
        let dt = *world.read_resource::<DeltaTime>();
        self.imgui
            .io_mut()
            .update_delta_time(std::time::Duration::from(dt));

        let mouse = self.imgui.io().want_capture_mouse;
        let keyboard = self.imgui.io().want_capture_keyboard;
        let text = self.imgui.io().want_text_input;

        let input_ctx = Self::create_input_context(mouse, keyboard, text)
            .expect("Failed to create inputo context for ui");

        *world
            .write_storage::<input::InputContext>()
            .get_mut(self.input_entity)
            .unwrap() = input_ctx;
    }

    fn forward_input(&mut self, world: &World) {
        use crate::io::input::{Input, MappedInput, StateId};

        let mut mapped_inputs = world.write_storage::<MappedInput>();
        let mapped_input = mapped_inputs
            .get_mut(self.input_entity)
            .expect("Did not find a mapped input for the ui entity");

        assert!(
            !self.imgui.io().want_set_mouse_pos,
            "Dear imgui wants to set mouse pos but this is not honored!"
        );

        let io = self.imgui.io_mut();
        io.keys_down = [false; 512];
        io.key_shift = false;
        io.key_ctrl = false;
        io.key_alt = false;
        io.key_super = false;
        io.mouse_down = [false; 5];
        for inp in mapped_input.drain().into_iter() {
            match inp {
                Input::State(state_id) if is_mouse_button(state_id) => {
                    log::debug!("imgui mouse button: {}", mouse_stateid_idx(state_id));
                    io.mouse_down[mouse_stateid_idx(state_id) as usize] = true
                }
                Input::State(StateId(key)) if is_keyboard_button(key) => {
                    use KeyCode::*;
                    io.keys_down[key as usize] = true;
                    io.key_shift = key == LShift as u32 || key == RShift as u32;
                    io.key_ctrl = key == LControl as u32 || key == RControl as u32;
                    io.key_alt = key == LAlt as u32 || key == RAlt as u32;
                    io.key_super = key == LWin as u32 || key == RWin as u32;
                }
                Input::Range(MOUSE_WHEEL_DELTA_Y, val) => io.mouse_wheel += val as f32,
                Input::Range(MOUSE_WHEEL_DELTA_X, val) => io.mouse_wheel_h += val as f32,
                Input::Text(chars) => {
                    for c in chars.iter() {
                        io.add_input_character(*c);
                    }
                }
                Input::CursorPos(pos) => {
                    log::debug!("imgui got cursor pos: {:?}", pos.0);
                    io.mouse_pos = [pos.x() as f32, pos.y() as f32];
                }
                i => unreachable!("{:?}", i),
            }
        }
    }

    pub fn build_ui<'a>(
        &mut self,
        world: &mut World,
        frame: &mut Frame<'a>,
    ) -> Option<UIDrawCommands> {
        log::trace!("Building ui");
        self.resize(frame.extent());
        self.forward_input(world);

        let ui = self.imgui.frame();
        crate::editor::build_ui(world, &ui);

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
        let vbuf_desc = OwningVertexBufferDescriptor::from_raw(
            vertices,
            ImGuiVertex::format(),
            BufferMutability::Mutable,
        );
        let ibuf_desc = OwningIndexBufferDescriptor::from_vec(indices, BufferMutability::Mutable);

        let (vertex_buffer, index_buffer) = if let Some(per_frame_data) = self.per_frame_data {
            frame
                .recreate_resource_blocking(per_frame_data.vertex_buffer, vbuf_desc)
                .expect("Bad vbuf handle");
            frame
                .recreate_resource_blocking(per_frame_data.index_buffer, ibuf_desc)
                .expect("Bad ibuf handle");
            (per_frame_data.vertex_buffer, per_frame_data.index_buffer)
        } else {
            let vh = frame
                .create_resource_blocking(vbuf_desc)
                .expect("Failed to create vertex buffer");
            let ih = frame
                .create_resource_blocking(ibuf_desc)
                .expect("Failed to create index buffer");

            (vh, ih)
        };

        let per_frame_data = PerFrameData {
            vertex_buffer,
            index_buffer,
            fb_width,
            fb_height,
        };

        self.per_frame_data = Some(per_frame_data);

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
    pub fn record_draw_commands<'b>(self, cmd_buf: &mut RenderPassEncoder<'b>) {
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
            .bind_push_constant(&pipeline, ShaderStage::VERTEX, &vertex_shader_data);

        for cmd in commands.iter() {
            cmd_buf.set_scissor(cmd.scissor).draw_indexed(
                cmd.count,
                cmd.indices_idx,
                cmd.vertices_idx,
            );
        }
    }
}
