use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use nalgebra_glm as glm;
use resurs::Handle;

use trekant::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, ShaderDescriptor, ShaderStage,
};
use trekant::util;
use trekant::vertex::{VertexDefinition, VertexFormat};
use trekant::{BufferDescriptor, BufferHandle, BufferMutability};
use trekant::{RenderPass, Renderer, Texture};

use trekant::Std140;

use std::time::{Duration, Instant};

const WINDOW_HEIGHT: u32 = 300;
const WINDOW_WIDTH: u32 = 300;
const WINDOW_TITLE: &str = "Trekanten Vulkan Tutorial";

struct State {
    pub window: winit::window::Window,
    frame_times: [Duration; 10],
    frame_time_idx: usize,
    start: Instant,
    frame_start: Instant,
}

impl State {
    pub fn new() -> (Self, EventLoop<()>) {
        let ev = EventLoop::new();
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title(WINDOW_TITLE)
            .build(&ev)
            .expect("Failed to build window");

        (
            Self {
                window,
                frame_times: [std::time::Duration::default(); 10],
                frame_time_idx: 0,
                start: Instant::now(),
                frame_start: Instant::now(),
            },
            ev,
        )
    }

    pub fn start_frame(&mut self) {
        self.frame_start = std::time::Instant::now();
    }

    pub fn end_frame(&mut self) {
        let time = std::time::Instant::now() - self.frame_start;
        self.frame_times[self.frame_time_idx] = time;

        if self.frame_time_idx < self.frame_times.len() - 1 {
            self.frame_time_idx += 1;
            return;
        }

        debug_assert!(self.frame_time_idx == (self.frame_times.len() - 1));

        let avg = self
            .frame_times
            .iter()
            .fold(Duration::from_secs(0), |acc, &t| acc + t)
            / self.frame_times.len() as u32;
        let s = format!(
            "{} (FPS: {:.2}, {:.2} ms)",
            WINDOW_TITLE,
            1.0 / avg.as_secs_f32(),
            1000.0 * avg.as_secs_f32()
        );
        self.window.set_title(&s);
        self.frame_time_idx = 0;
    }

    fn extents(&self) -> util::Extent2D {
        let PhysicalSize { width, height } = self.window.inner_size();
        util::Extent2D { width, height }
    }
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
struct Vertex {
    pos: [f32; 3],
    col: [f32; 3],
    tex_coord: [f32; 2],
}

unsafe impl VertexDefinition for Vertex {
    fn format() -> VertexFormat {
        VertexFormat::from([
            util::Format::FLOAT3,
            util::Format::FLOAT3,
            util::Format::FLOAT2,
        ])
    }
}

#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(transparent)]
struct Mat4([[f32; 4]; 4]);

unsafe impl trekant::Std140 for Mat4 {
    const SIZE: usize = 64;
    const ALIGNMENT: usize = 16;
}

#[derive(Clone, Copy, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

fn get_fname(dir: &str, target: &str) -> std::path::PathBuf {
    let url = reqwest::Url::parse(target).expect("Bad url");

    let fname = url
        .path_segments()
        .and_then(|segments| segments.last())
        .and_then(|name| if name.is_empty() { None } else { Some(name) })
        .unwrap_or("tmp.bin");

    std::env::current_dir().unwrap().join(dir).join(fname)
}

fn load_file(fname: &std::path::Path) -> std::io::Cursor<Vec<u8>> {
    use std::io::Read;
    let mut buf = Vec::new();
    let mut file = std::fs::File::open(fname).unwrap();
    file.read_to_end(&mut buf).unwrap();
    std::io::Cursor::new(buf)
}

fn load_url(dir: &str, target: &str) -> std::io::Cursor<Vec<u8>> {
    let fname = get_fname(dir, target);

    if fname.exists() {
        println!("File already exists: '{}'. Reusing!", fname.display());
    } else {
        println!("File to download: '{}'", fname.display());
        std::fs::create_dir_all(dir).expect("Failed to create_dir_all");
        let mut file = std::fs::File::create(&fname).expect("Failed to create file");
        let body = reqwest::blocking::get(target).unwrap().bytes().unwrap();
        std::io::copy(&mut body.as_ref(), &mut file).expect("failed to copy");
    }

    load_file(&fname)
}

const OBJ_URL: &str = "https://vulkan-tutorial.com/resources/viking_room.obj";
const TEX_URL: &str = "https://vulkan-tutorial.com/resources/viking_room.png";

static RAW_VERT_SPV: &[u32] = inline_spirv::include_spirv!(
    "examples/shaders/vulkan-tutorial.vert.glsl",
    vert,
    glsl,
    entry = "main"
);
static RAW_FRAG_SPV: &[u32] = inline_spirv::include_spirv!(
    "examples/shaders/vulkan-tutorial.frag.glsl",
    frag,
    glsl,
    entry = "main"
);

type VertexIndexTy = u32;

fn load_viking_house() -> (Vec<Vertex>, Vec<VertexIndexTy>) {
    let mut cursor = load_url("models", OBJ_URL);

    let (mut models, _) = tobj::load_obj_buf(&mut cursor, true, |_| {
        Ok((vec![], std::collections::HashMap::new()))
    })
    .unwrap();

    println!("# of models: {}", models.len());
    let model = models.remove(0);

    let mut vertices = Vec::new();

    let tobj::Model {
        mesh:
            tobj::Mesh {
                positions,
                texcoords,
                indices,
                ..
            },
        ..
    } = model;

    vertices.reserve(positions.len() / 3);
    for (pos, tc) in positions.chunks(3).zip(texcoords.chunks(2)) {
        let vertex = Vertex {
            pos: [pos[0], pos[1], pos[2]],
            col: [1.0, 1.0, 1.0],
            tex_coord: [tc[0], 1.0 - tc[1]],
        };
        vertices.push(vertex);
    }

    assert!(!vertices.is_empty());
    assert!(!indices.is_empty());

    (vertices, indices)
}

fn get_next_mvp(start: &std::time::Instant, aspect_ratio: f32) -> UniformBufferObject {
    let time = std::time::Instant::now() - *start;
    let time = time.as_secs_f32();

    let mut ubo = UniformBufferObject {
        model: Mat4(
            glm::rotate(
                &glm::identity(),
                time * std::f32::consts::FRAC_PI_2,
                &glm::vec3(0.0, 0.0, 1.0),
            )
            .into(),
        ),
        view: Mat4(
            glm::look_at(
                &glm::vec3(2.0, 2.0, 2.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 0.0, 1.0),
            )
            .into(),
        ),
        proj: Mat4(
            glm::perspective_zo(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0).into(),
        ),
    };

    ubo.proj.0[1][1] *= -1.0;

    ubo
}

fn create_texture(renderer: &mut Renderer) -> Handle<Texture> {
    let _ = load_url("textures", TEX_URL);
    let tex_path = get_fname("textures", TEX_URL);
    renderer
        .create_texture(trekant::TextureDescriptor::File {
            path: tex_path,
            format: util::Format::RGBA_SRGB,
            mipmaps: trekant::MipMaps::Generate,
            ty: trekant::TextureType::Tex2D,
        })
        .expect("Failed to create texture")
}

type Mesh = (BufferHandle, BufferHandle);

fn create_mesh(renderer: &mut Renderer) -> Mesh {
    let (vertices, indices) = load_viking_house();

    let vertex_buffer_descriptor =
        BufferDescriptor::vertex_buffer(&vertices, BufferMutability::Immutable);

    let vertex_buffer = renderer
        .create_buffer(vertex_buffer_descriptor)
        .expect("Failed to create vertex buffer");

    let index_buffer_descriptor =
        BufferDescriptor::index_buffer(&indices, BufferMutability::Immutable);
    let index_buffer = renderer
        .create_buffer(index_buffer_descriptor)
        .expect("Failed to create index buffer");
    (vertex_buffer, index_buffer)
}

fn create_mvp_ubuf(renderer: &mut Renderer) -> BufferHandle {
    let data = UniformBufferObject {
        model: Mat4::default(),
        view: Mat4::default(),
        proj: Mat4::default(),
    };

    let uniform_buffer_desc = BufferDescriptor::uniform_buffer(
        std::slice::from_ref(&data),
        BufferMutability::Mutable,
        trekant::BufferLayout::Std140,
    );

    renderer
        .create_buffer(uniform_buffer_desc)
        .expect("Failed to create uniform buffer")
}

fn create_pipeline(
    renderer: &mut Renderer,
    render_pass: &Handle<RenderPass>,
) -> Handle<GraphicsPipeline> {
    let vert = ShaderDescriptor {
        debug_name: Some("vulkan-tutorial-vert".to_owned()),
        spirv_code: RAW_VERT_SPV.to_vec(),
    };
    let frag = ShaderDescriptor {
        debug_name: Some("vulkan-tutorial-frag".to_owned()),
        spirv_code: RAW_FRAG_SPV.to_vec(),
    };
    let pipeline_descriptor = GraphicsPipelineDescriptor::builder()
        .vert(vert)
        .frag(frag)
        .vertex_format(Vertex::format())
        .build()
        .expect("Failed to build graphics pipeline descriptor");

    renderer
        .create_gfx_pipeline(pipeline_descriptor, render_pass)
        .expect("Failed to create graphics pipeline")
}

fn main() {
    env_logger::init();

    let (mut state, event_loop) = State::new();
    let mut renderer =
        trekant::Renderer::new(&state.window, state.extents()).expect("Failed to create renderer");

    let render_pass = renderer
        .presentation_render_pass(4)
        .expect("Failed to create render pass");

    let (vertex_buffer, index_buffer) = create_mesh(&mut renderer);
    let gfx_pipeline_handle = create_pipeline(&mut renderer, &render_pass);
    let uniform_buffer_handle = create_mvp_ubuf(&mut renderer);
    let texture_handle = create_texture(&mut renderer);
    let desc_set_handle = trekant::PipelineResourceSet::builder(&mut renderer)
        .add_buffer(uniform_buffer_handle, 0, ShaderStage::VERTEX)
        .add_texture(texture_handle, 1, ShaderStage::FRAGMENT, false)
        .build();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                state.start_frame();
                let aspect_ratio = renderer.aspect_ratio();
                let mut frame = match renderer.next_frame() {
                    Err(trekant::RenderError::NeedsResize(reason)) => {
                        log::info!("Resize reason: {:?}", reason);
                        renderer.resize(state.extents()).expect("Failed to resize");
                        renderer.next_frame()
                    }
                    x => x,
                }
                .expect("Failed to start frame");

                let next_mvp = get_next_mvp(&state.start, aspect_ratio);

                frame
                    .write_buffer_element(uniform_buffer_handle, &next_mvp, 0)
                    .expect("Failed to update uniform buffer");

                let cmd_buf = frame
                    .new_command_buffer()
                    .expect("Failed to build render command buffer");

                let mut builder = frame
                    .begin_presentation_pass(cmd_buf, &render_pass)
                    .expect("Failed to begin render pass");

                builder
                    .bind_graphics_pipeline(&gfx_pipeline_handle)
                    .bind_shader_resource_group(0, &desc_set_handle, &gfx_pipeline_handle)
                    .draw_mesh(vertex_buffer, index_buffer);

                let cmd_buf = builder.end().expect("Failed to end render command buffer");

                frame.add_command_buffer(cmd_buf);

                let frame = frame.finish();
                renderer
                    .submit(frame)
                    .or_else(|e| {
                        if let trekant::RenderError::NeedsResize(reason) = e {
                            log::info!("Resize reason: {:?}", reason);
                            renderer.resize(state.extents())
                        } else {
                            Err(e)
                        }
                    })
                    .expect("Failed to submit frame");
                state.end_frame();
            }
            _ => (),
        }
    });
}
