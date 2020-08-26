use glfw::{Action, Key};

use ash::vk;

use nalgebra_glm as glm;

use trekanten::mesh;
use trekanten::pipeline;
use trekanten::texture;
use trekanten::uniform;
use trekanten::window::Window;
use trekanten::Handle;
use trekanten::ResourceManager;

#[repr(C, packed)]
struct Vertex {
    pos: glm::Vec3,
    col: glm::Vec3,
    tex_coord: glm::Vec2,
}

impl trekanten::vertex::VertexDefinition for Vertex {
    fn binding_description() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn attribute_description() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: memoffset::offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: memoffset::offset_of!(Vertex, col) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: memoffset::offset_of!(Vertex, tex_coord) as u32,
            },
        ]
    }
}

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

// TODO:
// * Handle window requested resize
// * Wait while minimized
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

    let (vertices, indices) = load_viking_house();

    let mut window = trekanten::window::GlfwWindow::new();
    let mut renderer = trekanten::Renderer::new(&window, window.extents())?;

    let vertex_buffer_descriptor = mesh::VertexBufferDescriptor::from_slice(&vertices);
    let vertex_buffer_handle: Handle<mesh::VertexBuffer> = renderer
        .create_resource(vertex_buffer_descriptor)
        .expect("Failed to create vertex buffer");

    let index_buffer_descriptor = mesh::IndexBufferDescriptor::from_slice(&indices);
    let index_buffer_handle = renderer
        .create_resource(index_buffer_descriptor)
        .expect("Failed to create index buffer");

    let pipeline_descriptor = pipeline::GraphicsPipelineDescriptor::builder()
        .vertex_shader("vert.spv")
        .fragment_shader("frag.spv")
        .vertex_type::<Vertex>()
        .build()
        .expect("Failed to create graphics pipeline desc");

    let gfx_pipeline_handle = renderer
        .create_resource(pipeline_descriptor)
        .expect("Failed to create graphics pipeline");

    let uniform_buffer_desc =
        uniform::UniformBufferDescriptor::uninitialized::<UniformBufferObject>(1);

    let uniform_buffer_handle = renderer
        .create_resource(uniform_buffer_desc)
        .expect("Failed to create uniform buffer");

    let _ = load_url("textures", TEX_URL);
    let tex_path = get_fname("textures", TEX_URL);
    let texture_handle = renderer
        .create_resource(texture::TextureDescriptor::new(tex_path.into()))
        .expect("Failed to create texture");

    let desc_set_handle = renderer
        .create_descriptor_set(
            &gfx_pipeline_handle,
            &uniform_buffer_handle,
            &texture_handle,
        )
        .expect("Failed to create descriptor set");

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

        let mut frame = match renderer.next_frame() {
            Err(trekanten::RenderError::NeedsResize(reason)) => {
                log::info!("Resize reason: {:?}", reason);
                renderer.resize(window.extents())?;
                renderer.next_frame()
            }
            x => x,
        }?;

        let next_mvp = get_next_mvp(&start, renderer.aspect_ratio());
        renderer
            .update_uniform(&uniform_buffer_handle, &next_mvp)
            .expect("Failed to update uniform buffer!");

        let render_pass = renderer.render_pass();
        let extent = renderer.swapchain_extent();
        let framebuffer = renderer.framebuffer(&frame);

        let gfx_pipeline = renderer
            .get_resource(&gfx_pipeline_handle)
            .expect("Missing graphics pipeline");
        let index_buffer = renderer
            .get_resource(&index_buffer_handle)
            .expect("Missing index buffer");
        let vertex_buffer = renderer
            .get_resource(&vertex_buffer_handle)
            .expect("Missing vertex buffer");
        let desc_set = renderer
            .get_descriptor_set(&desc_set_handle)
            .expect("Missing descriptor set");

        let cmd_buf = frame
            .new_command_buffer()?
            .begin_render_pass(render_pass, framebuffer, extent)
            .bind_graphics_pipeline(&gfx_pipeline)
            .bind_descriptor_set(&desc_set, &gfx_pipeline)
            .bind_index_buffer(&index_buffer)
            .bind_vertex_buffer(&vertex_buffer)
            .draw_indexed(indices.len() as u32)
            .end_render_pass()
            .end()?;

        frame.add_command_buffer(cmd_buf);

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
