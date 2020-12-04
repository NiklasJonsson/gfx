use glfw::{Action, Key};
use nalgebra_glm as glm;
use resurs::Handle;

use trekanten::descriptor::DescriptorSet;
use trekanten::mesh;
use trekanten::pipeline;
use trekanten::pipeline::ShaderDescriptor;
use trekanten::pipeline::ShaderStage;
use trekanten::texture;
use trekanten::uniform;
use trekanten::util;
use trekanten::vertex::{VertexDefinition, VertexFormat};

use trekanten::loader::ResourceLoader as _;
use trekanten::ResourceManager as _;

use std::time::Duration;

const WINDOW_HEIGHT: u32 = 300;
const WINDOW_WIDTH: u32 = 300;
const WINDOW_TITLE: &str = "Vulkan Tutorial";

type GlfwWindowEvents = std::sync::mpsc::Receiver<(f64, glfw::WindowEvent)>;

struct GlfwWindow {
    pub glfw: glfw::Glfw,
    pub window: glfw::Window,
    pub events: GlfwWindowEvents,
    frame_times: [std::time::Duration; 10],
    frame_time_idx: usize,
}

impl GlfwWindow {
    pub fn new() -> Self {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).expect("Failed to init glfw");
        assert!(glfw.vulkan_supported(), "No vulkan!");

        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

        let (mut window, events) = glfw
            .create_window(
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                WINDOW_TITLE,
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create GLFW window.");

        window.set_key_polling(true);

        Self {
            glfw,
            window,
            events,
            frame_times: [std::time::Duration::default(); 10],
            frame_time_idx: 0,
        }
    }

    pub fn set_frame_ms(&mut self, time: Duration) {
        self.frame_times[self.frame_time_idx] = time;

        if self.frame_time_idx == self.frame_times.len() - 1 {
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
        } else {
            self.frame_time_idx += 1;
        }
    }

    fn extents(&self) -> util::Extent2D {
        let (w, h) = self.window.get_framebuffer_size();
        util::Extent2D {
            width: w as u32,
            height: h as u32,
        }
    }
}

unsafe impl raw_window_handle::HasRawWindowHandle for GlfwWindow {
    fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
        self.window.raw_window_handle()
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
        VertexFormat::builder()
            .add_attribute(util::Format::FLOAT3)
            .add_attribute(util::Format::FLOAT3)
            .add_attribute(util::Format::FLOAT2)
            .build()
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    proj: glm::Mat4,
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
    let mut file = std::fs::File::open(&fname).unwrap();
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

static RAW_VERT_SPV: &'static [u32] = inline_spirv::include_spirv!(
    "examples/shaders/shader.vert.glsl",
    vert,
    glsl,
    entry = "main"
);
static RAW_FRAG_SPV: &'static [u32] = inline_spirv::include_spirv!(
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

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        _ => {}
    }
}

fn get_next_mvp(start: &std::time::Instant, aspect_ratio: f32) -> UniformBufferObject {
    let time = std::time::Instant::now() - *start;
    let time = time.as_secs_f32();

    let mut ubo = UniformBufferObject {
        model: glm::rotate(
            &glm::identity(),
            time * std::f32::consts::FRAC_PI_2,
            &glm::vec3(0.0, 0.0, 1.0),
        ),
        view: glm::look_at(
            &glm::vec3(2.0, 2.0, 2.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 0.0, 1.0),
        ),
        proj: glm::perspective_zo(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0),
    };

    ubo.proj[(1, 1)] *= -1.0;

    ubo
}

fn main() -> Result<(), trekanten::RenderError> {
    env_logger::init();

    let mut window = GlfwWindow::new();
    let mut renderer = trekanten::Renderer::new(&window, window.extents())?;
    let loader = renderer.loader();

    // TODO:
    // * Pipeline should be async as well
    // * Per-frame uniform buffer updates should use frame resource commands
    // * Batch load
    // * drawlist to render pass
    // * figure out nice api for creating ubo & modifying every frame
    // * async error propagation

    // Use a channel to signal that it is done
    let (mesh_sender, mesh_receiver) = std::sync::mpsc::channel();
    let loader_clone = loader.clone();
    std::thread::spawn(move || {
        let (vertices, indices) = load_viking_house();

        let vertex_buffer_descriptor = mesh::OwningVertexBufferDescriptor::from_vec(
            vertices,
            mesh::BufferMutability::Immutable,
        );

        let vertex_buffer = loader_clone.load(vertex_buffer_descriptor);

        let index_buffer_descriptor =
            mesh::OwningIndexBufferDescriptor::from_vec(indices, mesh::BufferMutability::Immutable);
        let index_buffer = loader_clone.load(index_buffer_descriptor);

        let mesh = mesh::Mesh {
            vertex_buffer,
            index_buffer,
        };

        mesh_sender.send(mesh).expect("Failed to send");
    });

    let (texture_sender, texture_receiver) = std::sync::mpsc::channel();
    let loader_clone = loader.clone();
    std::thread::spawn(move || {
        let _ = load_url("textures", TEX_URL);
        let tex_path = get_fname("textures", TEX_URL);
        let descriptor = texture::TextureDescriptor::file(
            tex_path.into(),
            util::Format::RGBA_SRGB,
            texture::MipMaps::Generate,
        );
        let texture = loader_clone.load(descriptor);
        texture_sender.send(texture).expect("Failed to send");
    });

    let mut mesh: Option<mesh::Mesh> = None;
    let mut desc_set_handle: Option<Handle<DescriptorSet>> = None;
    let mut texture_handle: Option<Handle<texture::Texture>> = None;

    let vert = ShaderDescriptor::FromRawSpirv(RAW_VERT_SPV.to_vec());
    let frag = ShaderDescriptor::FromRawSpirv(RAW_FRAG_SPV.to_vec());
    let pipeline_descriptor = pipeline::GraphicsPipelineDescriptor::builder()
        .vert(vert)
        .frag(frag)
        .vertex_format(Vertex::format())
        .build()
        .expect("Failed to build graphics pipeline descriptor");

    let gfx_pipeline_handle = renderer
        .create_resource_blocking(pipeline_descriptor)
        .expect("Failed to create graphics pipeline");

    let data = vec![UniformBufferObject {
        model: glm::Mat4::default(),
        view: glm::Mat4::default(),
        proj: glm::Mat4::default(),
    }];

    let uniform_buffer_desc =
        uniform::OwningUniformBufferDescriptor::from_vec(data, mesh::BufferMutability::Mutable);

    let uniform_buffer_handle = renderer
        .create_resource_blocking(uniform_buffer_desc)
        .expect("Failed to create uniform buffer");

    let start = std::time::Instant::now();
    let mut last = start;
    while !window.window.should_close() {
        let now = std::time::Instant::now();
        let diff = now - last;
        window.set_frame_ms(diff);
        last = now;

        window.glfw.poll_events();
        for (_, event) in glfw::flush_messages(&window.events) {
            handle_window_event(&mut window.window, event);
        }

        mesh = mesh.or(mesh_receiver.try_recv().ok());
        texture_handle = texture_handle.or(texture_receiver.try_recv().ok());

        if let Some(texture_handle) = texture_handle {
            let is_pending = renderer
                .get_resource(&texture_handle)
                .expect("This should have been inserted...")
                .is_pending();
            if !is_pending && desc_set_handle.is_none() {
                desc_set_handle = Some(
                    DescriptorSet::builder(&mut renderer)
                        .add_buffer(&uniform_buffer_handle, 0, ShaderStage::Vertex)
                        .add_texture(&texture_handle, 1, ShaderStage::Fragment)
                        .build(),
                );
            }
        }

        let aspect_ratio = renderer.aspect_ratio();

        let mut frame = match renderer.next_frame() {
            Err(trekanten::RenderError::NeedsResize(reason)) => {
                log::info!("Resize reason: {:?}", reason);
                renderer.resize(window.extents())?;
                renderer.next_frame()
            }
            x => x,
        }?;

        let next_mvp = get_next_mvp(&start, aspect_ratio);
        frame
            .update_uniform_blocking(&uniform_buffer_handle, &next_mvp)
            .expect("Failed to update uniform buffer!");

        // Check if loading is done and gpu load has been submitted
        if let (Some(mesh), Some(desc_set)) = (&mesh, &desc_set_handle) {
            // Pass the handles to the render pass and the renderer will ignore them if they haven't loaded yet
            let mut builder = frame
                .begin_render_pass()
                .expect("Failed to begin render pass");

            builder
                .bind_graphics_pipeline(&gfx_pipeline_handle)
                .bind_shader_resource_group(0, &desc_set, &gfx_pipeline_handle)
                .draw_mesh(mesh);

            let cmd_buf = builder
                .build()
                .expect("Failed to build render command buffer");
            frame.add_raw_command_buffer(cmd_buf);
        }

        let frame = frame.finish();
        renderer.submit(frame).or_else(|e| {
            if let trekanten::RenderError::NeedsResize(reason) = e {
                log::info!("Resize reason: {:?}", reason);
                renderer.resize(window.extents())
            } else {
                Err(e)
            }
        })?;
    }

    Ok(())
}
