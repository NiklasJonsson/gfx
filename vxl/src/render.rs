use trekanten::buffer::{DeviceIndexBuffer, DeviceVertexBuffer};
use trekanten::descriptor::DescriptorSet;
use trekanten::pipeline::{
    GraphicsPipeline, GraphicsPipelineDescriptor, ShaderDescriptor, ShaderStage,
};
use trekanten::{buffer, util};
use trekanten::{BufferHandle, Handle, RenderPass, Std140Compat};

use std::error::Error;
use std::path::Path;

#[derive(Clone, Copy, Std140Compat)]
struct ViewData {
    view_proj: [f32; 16],
    view_pos: [f32; 4],
}

#[derive(Clone, Copy, Std140Compat)]
struct Model {
    model: [f32; 16],
    model_it: [f32; 16],
}

// TODO: Share math code
pub fn perspective_vk(
    fov_y_radians: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
) -> vek::Mat4<f32> {
    let mut m = vek::Mat4::perspective_rh_zo(fov_y_radians, aspect_ratio, near, far);
    // vulkan has the y-axis inverted (right-handed upside-down).
    m[(1, 1)] *= -1.0;

    m
}

fn next_viewdata(aspect_ratio: f32) -> ViewData {
    let eye = vek::Vec4::new(1_f32, 0., 1., 1.);
    let target = vek::Vec4::new(2_f32, 0., 2., 1.);
    let view = vek::Mat4::<f32>::look_at_rh(eye, target, vek::Vec4::unit_y());
    let proj = perspective_vk(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.1, 100.0);
    ViewData {
        view_proj: (proj * view).into_col_array(),
        view_pos: todo!("Get position from matrix"),
    }
}

fn compile_shader(
    compiler: &mut shaderc::Compiler,
    path: &Path,
    shader_kind: shaderc::ShaderKind,
    options: Option<&shaderc::CompileOptions>,
) -> Result<shaderc::CompilationArtifact, shaderc::Error> {
    let vert =
        std::fs::read_to_string(path).expect(&format!("Failed to read file {}", path.display()));
    compiler.compile_into_spirv(
        &vert,
        shader_kind,
        path.to_str().unwrap_or("INVALID_SHADER_PATH"),
        "main",
        options,
    )
}

fn shader_compile_error(err: shaderc::Error) {
    use shaderc::Error;

    match err {
       Error::CompilationError(count, msg) => {
        println!("Compilation errors: {count}");
        for line in msg.lines() {
            println!("{}", line);
        }
       }
       e => panic!("{}", e.to_string()),
    }
}

fn create_pipeline(
    renderer: &mut trekanten::Renderer,
    compiler: &mut shaderc::Compiler,
    render_pass: &Handle<RenderPass>,
) -> Handle<GraphicsPipeline> {
    use trekanten::vertex::VertexDefinition as _;
    let vert = compile_shader(
        compiler,
        Path::new("vxl/src/shaders/vert.glsl"),
        shaderc::ShaderKind::Vertex,
        None,
    )
    .map_err(shader_compile_error)
    .expect("Failed to compile vert shader");
    let frag = compile_shader(
        compiler,
        Path::new("vxl/src/shaders/frag.glsl"),
        shaderc::ShaderKind::Fragment,
        None,
    )
    .map_err(shader_compile_error)
    .expect("Failed to compile frag shader");

    let vert = ShaderDescriptor::FromRawSpirv(vert.as_binary().to_vec());
    let frag = ShaderDescriptor::FromRawSpirv(frag.as_binary().to_vec());

    let pipeline_descriptor = GraphicsPipelineDescriptor::builder()
        .vert(vert)
        .frag(frag)
        .vertex_format(crate::meshing::Vertex::format())
        .build()
        .expect("Failed to build graphics pipeline descriptor");

    renderer
        .create_gfx_pipeline(pipeline_descriptor, render_pass)
        .expect("Failed to create graphics pipeline")
}

pub struct Rendering {
    pub renderer: trekanten::Renderer,
    render_pass: Handle<RenderPass>,
    viewdata: buffer::BufferHandle<buffer::DeviceUniformBuffer>,
    gfx_pipeline: Handle<GraphicsPipeline>,
    desc_set: Handle<DescriptorSet>,
}

#[derive(Clone, Copy)]
pub struct RenderCmd {
    pub vertices: BufferHandle<DeviceVertexBuffer>,
    pub indices: BufferHandle<DeviceIndexBuffer>,
}

impl Rendering {
    pub fn new<W>(window: &W, window_extent: util::Extent2D) -> Result<Self, trekanten::RenderError>
    where
        W: raw_window_handle::HasRawWindowHandle,
    {
        let mut renderer = trekanten::Renderer::new(window, window_extent)?;
        let mut compiler = shaderc::Compiler::new().expect("Failed to create shaderc");

        let render_pass = renderer.presentation_render_pass(4)?;

        let viewdata = create_viewdata_ubuf(&mut renderer);
        let gfx_pipeline = create_pipeline(&mut renderer, &mut compiler, &render_pass);

        let desc_set = trekanten::descriptor::DescriptorSet::builder(&mut renderer)
            .add_buffer(&viewdata, 0, ShaderStage::VERTEX)
            .build();

        Ok(Self {
            viewdata,
            render_pass,
            renderer,
            gfx_pipeline,
            desc_set,
        })
    }
    pub fn render(&mut self, window_extents: util::Extent2D, cmds: &[RenderCmd]) {
        let aspect_ratio = self.renderer.aspect_ratio();
        let mut frame = match self.renderer.next_frame() {
            Err(trekanten::RenderError::NeedsResize(reason)) => {
                log::info!("Resize reason: {:?}", reason);
                self.renderer
                    .resize(window_extents)
                    .expect("Failed to resize");
                self.renderer.next_frame()
            }
            x => x,
        }
        .expect("Failed to start frame");

        let next_viewdata = next_viewdata(aspect_ratio);
        frame
            .update_uniform_blocking(&self.viewdata, &next_viewdata)
            .expect("Failed to update uniform buffer!");
        let cmd_buf = frame
            .new_command_buffer()
            .expect("Failed to build render command buffer");

        let mut builder = frame
            .begin_presentation_pass(cmd_buf, &self.render_pass)
            .expect("Failed to begin render pass");

        builder
            .bind_graphics_pipeline(&self.gfx_pipeline)
            .bind_shader_resource_group(0, &self.desc_set, &self.gfx_pipeline);

        for cmd in cmds {
            builder.draw_mesh(&cmd.vertices, &cmd.indices);
        }

        let cmd_buf = builder.end().expect("Failed to end render command buffer");

        frame.add_command_buffer(cmd_buf);

        let frame = frame.finish();
        self.renderer
            .submit(frame)
            .or_else(|e| {
                if let trekanten::RenderError::NeedsResize(reason) = e {
                    log::info!("Resize reason: {:?}", reason);
                    self.renderer.resize(window_extents)
                } else {
                    Err(e)
                }
            })
            .expect("Failed to submit frame");
    }
}

fn create_viewdata_ubuf(
    renderer: &mut trekanten::Renderer,
) -> buffer::BufferHandle<buffer::DeviceUniformBuffer> {
    use trekanten::ResourceManager as _;

    let data = vec![ViewData {
        view_proj: Default::default(),
        view_pos: Default::default(),
    }];

    let uniform_buffer_desc =
        buffer::UniformBufferDescriptor::from_slice(&data, buffer::BufferMutability::Mutable);

    renderer
        .create_resource_blocking(uniform_buffer_desc)
        .expect("Failed to create uniform buffer")
}
