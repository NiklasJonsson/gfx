use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use nalgebra_glm as glm;
use resurs::{Async, Handle};

use buffer::{
    BufferHandle, BufferMutability, DeviceIndexBuffer, DeviceUniformBuffer, DeviceVertexBuffer,
};
use trekant::buffer;
use trekant::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, ShaderDescriptor, ShaderStage,
};
use trekant::util;
use trekant::vertex::{VertexDefinition, VertexFormat};

use trekant::Std140Compat;
use trekant::{Loader, RenderPass, Renderer, Texture};

use trekant::loader::ResourceLoader as _;
use trekant::ResourceManager as _;

use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WINDOW_HEIGHT: u32 = 300;
const WINDOW_WIDTH: u32 = 300;
const WINDOW_TITLE: &str = "Trekanten Vulkan Tutoria";

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

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct Vertex {
    pos: glm::Vec3,
    col: glm::Vec3,
    tex_coord: glm::Vec2,
}

impl VertexDefinition for Vertex {
    fn format() -> VertexFormat {
        VertexFormat::from([
            util::Format::FLOAT3,
            util::Format::FLOAT3,
            util::Format::FLOAT2,
        ])
    }
}

#[derive(Clone, Copy, Default)]
#[repr(transparent)]
struct Mat4(glm::Mat4);

#[derive(Clone, Copy, Std140Compat)]
#[repr(C)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

unsafe impl trekant::Std140 for Mat4 {
    const SIZE: usize = 64;
    const ALIGNMENT: usize = 16;
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
    "examples/shaders/shader.vert.glsl",
    vert,
    glsl,
    entry = "main"
);
static RAW_FRAG_SPV: &[u32] = inline_spirv::include_spirv!(
    "examples/shaders/shader.frag.glsl",
    frag,
    glsl,
    entry = "main"
);

fn load_viking_house() -> (Vec<Vertex>, Vec<u32>) {
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

    for (pos, tc) in positions.chunks(3).zip(texcoords.chunks(2)) {
        let vertex = Vertex {
            pos: glm::vec3(pos[0], pos[1], pos[2]),
            col: glm::vec3(1.0, 1.0, 1.0),
            tex_coord: glm::vec2(tc[0], 1.0 - tc[1]),
        };
        vertices.push(vertex);
    }

    (vertices, indices)
}

fn get_next_mvp(start: &std::time::Instant, aspect_ratio: f32) -> UniformBufferObject {
    let time = std::time::Instant::now() - *start;
    let time = time.as_secs_f32();

    let mut ubo = UniformBufferObject {
        model: Mat4(glm::rotate(
            &glm::identity(),
            time * std::f32::consts::FRAC_PI_2,
            &glm::vec3(0.0, 0.0, 1.0),
        )),
        view: Mat4(glm::look_at(
            &glm::vec3(2.0, 2.0, 2.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 0.0, 1.0),
        )),
        proj: Mat4(glm::perspective_zo(
            aspect_ratio,
            std::f32::consts::FRAC_PI_4,
            0.1,
            10.0,
        )),
    };

    ubo.proj.0[(1, 1)] *= -1.0;

    ubo
}

fn create_texture(
    loader: Arc<Loader>,
    texture_sender: mpsc::Sender<Handle<Async<trekant::Texture>>>,
) {
    let _ = load_url("textures", TEX_URL);
    let tex_path = get_fname("textures", TEX_URL);
    let descriptor = trekant::TextureDescriptor::File {
        path: tex_path,
        format: util::Format::RGBA_SRGB,
        mipmaps: trekant::MipMaps::None,
    };

    let texture = loader
        .load_texture(descriptor)
        .expect("Failed to load texture");
    texture_sender.send(texture).expect("Failed to send");
}

type PendingMesh = (
    BufferHandle<Async<DeviceVertexBuffer>>,
    BufferHandle<Async<DeviceIndexBuffer>>,
);

fn create_mesh(loader: Arc<Loader>, mesh_sender: mpsc::Sender<PendingMesh>) {
    let (vertices, indices) = load_viking_house();

    let vertex_buffer_descriptor =
        buffer::VertexBufferDescriptor::from_vec(vertices, BufferMutability::Immutable);

    let vertex_buffer = loader
        .load(vertex_buffer_descriptor)
        .expect("Failed to load vertex buffer");

    let index_buffer_descriptor =
        buffer::IndexBufferDescriptor::from_vec(indices, BufferMutability::Immutable);
    let index_buffer = loader
        .load(index_buffer_descriptor)
        .expect("Failed to load index buffer");

    mesh_sender
        .send((vertex_buffer, index_buffer))
        .expect("Failed to send");
}

fn create_mvp_ubuf(renderer: &mut Renderer) -> BufferHandle<DeviceUniformBuffer> {
    let data = vec![UniformBufferObject {
        model: Mat4::default(),
        view: Mat4::default(),
        proj: Mat4::default(),
    }];

    let uniform_buffer_desc =
        buffer::UniformBufferDescriptor::from_slice(&data, buffer::BufferMutability::Mutable);

    renderer
        .create_resource_blocking(uniform_buffer_desc)
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
    let loader = Arc::new(renderer.loader().expect("Loader should be available"));
    let render_pass = renderer
        .presentation_render_pass(4)
        .expect("Failed to create render pass");

    let (mesh_sender, mesh_receiver) = std::sync::mpsc::channel();
    let loader_clone = loader.clone();
    std::thread::spawn(move || create_mesh(loader_clone, mesh_sender));

    let (texture_sender, texture_receiver) = std::sync::mpsc::channel();
    let loader_clone = loader.clone();
    std::thread::spawn(move || create_texture(loader_clone, texture_sender));

    let gfx_pipeline_handle = create_pipeline(&mut renderer, &render_pass);
    let ubuf = create_mvp_ubuf(&mut renderer);

    // The call to load happens async so in the main-loop, we query the channel each frame if we are done.
    // After that, we still might need to wait for the gpu resource upload. When that is done, we will get
    // the final handle back with the transfer() call, ready to to be used when rendering.
    let mut pending_mesh: Option<PendingMesh> = None;
    let mut pending_tex: Option<Handle<Async<Texture>>> = None;

    // The final handles used for rendering
    let mut vbuf: Option<BufferHandle<DeviceVertexBuffer>> = None;
    let mut ibuf: Option<BufferHandle<DeviceIndexBuffer>> = None;
    let mut tex: Option<Handle<Texture>> = None;

    // Can only created once the texture is done.
    let mut desc_set: Option<Handle<trekant::PipelineResourceSet>> = None;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                state.start_frame();
                // Query the manually spawned threads
                pending_mesh = pending_mesh.or_else(|| mesh_receiver.try_recv().ok());
                pending_tex = pending_tex.or_else(|| texture_receiver.try_recv().ok());

                // It is fairly unlikely but it could happen that the gpu upload happens before the spawned threads have returned their data.
                // If that happens we wouldn't have the pending handles to compare with when transfering. In our case, it doesn't matter since
                // we only have one handle of each type, but in the general case, we need to have the async handles to map the incoming ones to.
                // We still want to wait for them though, because we need to have exclusive ownership
                if let (Some(pending_mesh), Some(pending_tex)) = (pending_mesh, pending_tex) {
                    let mut guard = loader.transfer(&mut renderer);
                    for mapping in guard.iter() {
                        use trekant::loader::HandleMapping;

                        match mapping {
                            HandleMapping::IndexBuffer { old, new } => {
                                assert_eq!(old.handle(), pending_mesh.1.handle());
                                assert!(ibuf.is_none(), "Double return?");
                                ibuf = Some(new);
                            }
                            HandleMapping::VertexBuffer { old, new } => {
                                assert_eq!(old.handle(), pending_mesh.0.handle());
                                assert!(vbuf.is_none(), "Double return?");
                                vbuf = Some(new);
                            }
                            HandleMapping::Texture { old, new } => {
                                assert_eq!(old, pending_tex);
                                assert!(tex.is_none());
                                tex = Some(new);
                            }
                            _ => unreachable!("Haven't tried to load any other types"),
                        }
                    }
                }

                // Create the descriptor set if we've received the texture
                if let (Some(tex), None) = (tex, desc_set) {
                    // We can't generate mipmaps as part of the loading via loader so we do it here.
                    renderer
                        .generate_mipmaps(&[tex])
                        .expect("Failed to generate mipmaps");
                    desc_set = Some(
                        trekant::PipelineResourceSet::builder(&mut renderer)
                            .add_buffer(&ubuf, 0, ShaderStage::VERTEX)
                            .add_texture(&tex, 1, ShaderStage::FRAGMENT, false)
                            .build(),
                    );
                }

                if vbuf.is_none() && ibuf.is_none() && desc_set.is_none() {
                    // Nothing to render
                    state.end_frame();
                    return;
                }

                let aspect_ratio = renderer.aspect_ratio();

                let mut frame = match renderer.next_frame() {
                    Err(trekant::RenderError::NeedsResize(reason)) => {
                        log::info!("Resize reason: {:?}", reason);
                        renderer.resize(state.extents()).expect("Failed to resize");
                        renderer.next_frame()
                    }
                    x => x,
                }
                .expect("Failed to get next frame");

                let next_mvp = get_next_mvp(&state.start, aspect_ratio);
                frame
                    .update_uniform_blocking(&ubuf, &next_mvp)
                    .expect("Failed to update uniform buffer!");

                // All loading needs to be done before drawing
                if let (Some(vbuf), Some(ibuf), Some(desc_set)) = (&vbuf, &ibuf, desc_set) {
                    let cmd_buf = frame
                        .new_command_buffer()
                        .expect("Failed to create render command buffer");
                    // Pass the handles to the render pass and the renderer will ignore them if they haven't loaded yet
                    let mut builder = frame
                        .begin_presentation_pass(cmd_buf, &render_pass)
                        .expect("Failed to begin render pass");

                    builder
                        .bind_graphics_pipeline(&gfx_pipeline_handle)
                        .bind_shader_resource_group(0, &desc_set, &gfx_pipeline_handle)
                        .draw_mesh(vbuf, ibuf);

                    let cmd_buf = builder.end().expect("Failed to end render command buffer");
                    frame.add_command_buffer(cmd_buf);
                }

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
