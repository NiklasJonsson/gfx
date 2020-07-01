use std::sync::Arc;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::device::Device;
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::shader::{GraphicsShaderType, ShaderModule};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract};

use std::ffi::CStr;
use std::fs::File;
use std::io::Read;

use crate::common::*;

pub mod vs_passthrough {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform Transforms {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 0, binding = 1) uniform Model {
    mat4 model;
    mat4 model_it;
} model_ubo;

layout(location = 0) in vec3 position;

void main() {
    gl_Position = ubo.proj * ubo.view * model_ubo.model * vec4(position, 1.0);
}
"
    }
}

pub mod fs_uniform_color {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform Color {
    vec4 x;
} color;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = color.x;
}
"
    }
}
pub mod pbr {
    pub mod vs {
        pub mod base {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/render/shaders/pbr_gltf_vert.glsl",
            }
        }

        pub mod uv {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/render/shaders/pbr_gltf_vert.glsl",
                define: [("HAS_TEX_COORDS", "1"),
                         ("TEX_COORDS_LOC", "2")],
            }
        }

        pub mod uv_vcol {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/render/shaders/pbr_gltf_vert.glsl",
                define: [("HAS_TEX_COORDS", "1"),
                          ("HAS_VERTEX_COLOR", "1"),
                          ("TEX_COORDS_LOC", "2"),
                          ("VCOL_LOC", "3")],
            }
        }

        pub mod uv_tan {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/render/shaders/pbr_gltf_vert.glsl",
                define: [("HAS_TEX_COORDS", "1"),
                          ("HAS_TANGENTS", "1"),
                          ("TEX_COORDS_LOC", "2"),
                          ("TAN_LOC", "3"),
                          ("BITAN_LOC", "4"),
                          ],
            }
        }

        pub mod uv_vcol_tan {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/render/shaders/pbr_gltf_vert.glsl",
                define: [("HAS_TEX_COORDS", "1"),
                          ("HAS_VERTEX_COLOR", "1"),
                          ("HAS_TANGENTS", "1"),
                          ("TEX_COORDS_LOC", "2"),
                          ("VCOL_LOC", "3"),
                          ("TAN_LOC", "4"),
                          ("BITAN_LOC", "5"),
                          ],
            }
        }
    }
    pub mod fs {
        pub mod base {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/render/shaders/pbr_gltf_frag.glsl",
            }
        }
        // Base color texture
        pub mod bc_tex {
            vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/render/shaders/pbr_gltf_frag.glsl",
            define: [("HAS_TEX_COORDS", "1"),
                      ("TEX_COORDS_LOC", "2"),
                      ("HAS_BASE_COLOR_TEXTURE", "1")],
            }
        }

        // Base color + metallic roughness texture + normal map
        pub mod bc_mr_nm_tex {
            vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/render/shaders/pbr_gltf_frag.glsl",
            define: [("HAS_TEX_COORDS", "1"),
                      ("HAS_TANGENTS", "1"),
                      ("TEX_COORDS_LOC", "2"),
                      ("TAN_LOC", "3"),
                      ("BITAN_LOC", "4"),

                      ("HAS_BASE_COLOR_TEXTURE", "1"),
                      ("HAS_METALLIC_ROUGHNESS_TEXTURE", "1"),
                      ("HAS_NORMAL_MAP", "1")],
            }
        }
        pub mod uv_vcol {
            vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/render/shaders/pbr_gltf_frag.glsl",
            define: [("HAS_TEX_COORDS", "1"),
                      ("HAS_VERTEX_COLOR", "1"),
                      ("TEX_COORDS_LOC", "2"),
                      ("VCOL_LOC", "3")],
            }
        }
    }
}

macro_rules! create_pipeline {
    ($device:ident, $builder:ident, $vert_mod:path, $frag_mod:path, $vert_type:ty, $render_mode:ident) => {{
        use $frag_mod as fmod;
        use $vert_mod as vmod;
        match $render_mode {
            CompilationMode::CompileTime => {
                let vs = vmod::Shader::load(Arc::clone($device))
                    .expect("Vertex shader compilation failed");
                let fs = fmod::Shader::load(Arc::clone($device))
                    .expect("Fragment shader compilation failed");

                Arc::new(
                    $builder
                        .vertex_input_single_buffer::<$vert_type>()
                        // How to interpret the vertex input
                        .vertex_shader(vs.main_entry_point(), ())
                        .fragment_shader(fs.main_entry_point(), ())
                        .build(Arc::clone($device))
                        .expect("Could not create graphics pipeline"),
                )
            }
            CompilationMode::RunTime { vs_path, fs_path } => {
                let vs = {
                    let mut f = File::open(vs_path).expect(
                        format!(
                            "Couldn't open runtime vert shader from path: {}",
                            vs_path.display()
                        )
                        .as_str(),
                    );
                    let mut v = vec![];
                    f.read_to_end(&mut v).unwrap();
                    unsafe { ShaderModule::new(Arc::clone($device), &v) }.unwrap()
                };

                let vs_main = unsafe {
                    vs.graphics_entry_point(
                        CStr::from_bytes_with_nul_unchecked(b"main\0"),
                        vmod::MainInput,
                        vmod::MainOutput,
                        vmod::Layout(ShaderStages {
                            vertex: true,
                            ..ShaderStages::none()
                        }),
                        GraphicsShaderType::Vertex,
                    )
                };

                let fs = {
                    let mut f = File::open(fs_path).expect(
                        format!(
                            "Couldn't open runtime frag shader from path: {}",
                            fs_path.display()
                        )
                        .as_str(),
                    );
                    let mut v = vec![];
                    f.read_to_end(&mut v).unwrap();
                    // Create a ShaderModule on a device the same Shader::load does it.
                    unsafe { ShaderModule::new(Arc::clone($device), &v) }.unwrap()
                };

                let fs_main = unsafe {
                    fs.graphics_entry_point(
                        CStr::from_bytes_with_nul_unchecked(b"main\0"),
                        fmod::MainInput,
                        fmod::MainOutput,
                        fmod::Layout(ShaderStages {
                            fragment: true,
                            ..ShaderStages::none()
                        }),
                        GraphicsShaderType::Fragment,
                    )
                };

                Arc::new(
                    $builder
                        .vertex_input_single_buffer::<$vert_type>()
                        // How to interpret the vertex input
                        .vertex_shader(vs_main, ())
                        .fragment_shader(fs_main, ())
                        .build(Arc::clone($device))
                        .expect("Could not create graphics pipeline"),
                )
            }
        }
    }};
}

pub fn create_graphics_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    swapchain_dimensions: [u32; 2],
    rendering_mode: PrimitiveTopology,
    vertex_buf: &VertexBuf,
    material: &Material,
    compilation_mode: &CompilationMode,
) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
    let dims = [
        swapchain_dimensions[0] as f32,
        swapchain_dimensions[1] as f32,
    ];

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: dims,
        depth_range: 0.0..1.0,
    };

    let builder = GraphicsPipeline::start()
        .primitive_topology(rendering_mode)
        // Whether to support special indices in in the vertex buffer to split triangles
        .primitive_restart(false)
        .viewports([viewport].iter().cloned())
        .depth_clamp(false)
        .polygon_mode_fill()
        .line_width(1.0)
        .cull_mode_back()
        .depth_stencil_simple_depth()
        .blend_pass_through()
        .front_face_counter_clockwise()
        .render_pass(Subpass::from(Arc::clone(render_pass), 0).unwrap());

    match (vertex_buf, material) {
        (VertexBuf::Base(_), Material::GlTFPBR { .. }) => create_pipeline!(
            device,
            builder,
            pbr::vs::base,
            pbr::fs::base,
            VertexBase,
            compilation_mode
        ),
        (
            VertexBuf::UV(_),
            Material::GlTFPBR {
                base_color_texture: Some(_),
                normal_map: None,
                metallic_roughness_texture: None,
                ..
            },
        ) => create_pipeline!(
            device,
            builder,
            pbr::vs::uv,
            pbr::fs::bc_tex,
            VertexUV,
            compilation_mode
        ),
        (
            VertexBuf::UVTan(_),
            Material::GlTFPBR {
                base_color_texture: Some(_),
                metallic_roughness_texture: Some(_),
                normal_map: Some(_),
                ..
            },
        ) => create_pipeline!(
            device,
            builder,
            pbr::vs::uv_tan,
            pbr::fs::bc_mr_nm_tex,
            VertexUVTan,
            compilation_mode
        ),
        (
            VertexBuf::UVCol(_),
            Material::GlTFPBR {
                base_color_texture: Some(_),
                normal_map: None,
                metallic_roughness_texture: None,
                ..
            },
        ) => create_pipeline!(
            device,
            builder,
            pbr::vs::uv_vcol,
            pbr::fs::uv_vcol,
            VertexUVCol,
            compilation_mode
        ),
        (VertexBuf::PosOnly(_), Material::Color { .. }) => create_pipeline!(
            device,
            builder,
            vs_passthrough,
            fs_uniform_color,
            VertexPosOnly,
            compilation_mode
        ),
        (_, _) => unimplemented!("Support more shader variants"),
    }
}
