use std::sync::Arc;
use vulkano::device::Device;
use vulkano::framebuffer::{RenderPassAbstract, Subpass};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::input_assembly::PrimitiveTopology;

use crate::common::*;

pub mod vs_static {
    vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform VPUniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 matrix;
} model;

layout(location = 0) in vec3 position;
layout(location = 2) in vec2 tex_coords;

layout(location = 1) out vec2 frag_tex_coords;

void main() {
    gl_Position = ubo.proj * ubo.view * model.matrix * vec4(position, 1.0);
    frag_tex_coords = tex_coords;
}
"
    }
}

pub mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform VPUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;


layout(location = 0) in vec3 position;
layout(location = 2) in vec2 tex_coords;

layout(location = 1) out vec2 frag_tex_coords;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    frag_tex_coords = tex_coords;
}
"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
    ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 1) in vec2 frag_tex_coords;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, frag_tex_coords);
}
"
    }
}

// TODO: Can we autogenerate these variants based on Vertex:s Optional members?
pub mod vs_pbr {
    vulkano_shaders::shader! {
    ty: "vertex",
    path: "src/render/shaders/pbr_gltf_vert.glsl",
    }
}

pub mod fs_pbr {
    vulkano_shaders::shader! {
    ty: "fragment",
    path: "src/render/shaders/pbr_gltf_frag.glsl",
    }
}

pub mod vs_pbr_base_color_texture {
    vulkano_shaders::shader! {
    ty: "vertex",
    path: "src/render/shaders/pbr_gltf_vert.glsl",
    define: &[("HAS_TEX_COORDS", "1"), ("HAS_BASE_COLOR_TEXTURE", "1")],
    }
}

pub mod fs_pbr_base_color_texture {
    vulkano_shaders::shader! {
    ty: "fragment",
    path: "src/render/shaders/pbr_gltf_frag.glsl",
    define: &[("HAS_TEX_COORDS", "1"), ("HAS_BASE_COLOR_TEXTURE", "1")],
    }
}

pub mod vs_pbr_uv_col {
    vulkano_shaders::shader! {
    ty: "vertex",
    path: "src/render/shaders/pbr_gltf_vert.glsl",
    define: &[("HAS_TEX_COORDS", "1"), ("HAS_VERTEX_COLOR", "1"), ],
    }
}

pub mod fs_pbr_uv_col {
    vulkano_shaders::shader! {
    ty: "fragment",
    path: "src/render/shaders/pbr_gltf_frag.glsl",
    define: &[("HAS_TEX_COORDS", "1"), ("HAS_VERTEX_COLOR", "1")],
    }
}

/*
 *
        Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<T>()
                // How to interpret the vertex input
                .vertex_shader(vs.main_entry_point(), ())
                .primitive_topology(rendering_mode)
                // Whether to support special indices in in the vertex buffer to split triangles
                .primitive_restart(false)
                .viewports([viewport].iter().cloned())
                .depth_clamp(false)
                .polygon_mode_fill()
                .line_width(1.0)
                .cull_mode_back()
                .depth_stencil_simple_depth()
                .fragment_shader(fs.main_entry_point(), ())
                .blend_pass_through()
                .front_face_counter_clockwise()
                .render_pass(Subpass::from(Arc::clone(render_pass), 0).unwrap())
                .build(Arc::clone(device))
                .expect("Could not create graphics pipeline"),
        )
*/

pub fn create_graphics_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPassAbstract + Send + Sync>,
    swapchain_dimensions: [u32; 2],
    rendering_mode: PrimitiveTopology,
    vertex_buf: &VertexBuf,
    material: &Material,
) -> Arc<GraphicsPipelineAbstract + Send + Sync>
{
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
        (VertexBuf::Base(_), _) => {
            let vs = vs_pbr::Shader::load(Arc::clone(device))
                .expect("Vertex shader compilation failed");
            let fs = fs_pbr::Shader::load(Arc::clone(device))
                .expect("Fragment shader compilation failed");

            Arc::new(
                builder
                    .vertex_input_single_buffer::<VertexBase>()
                    // How to interpret the vertex input
                    .vertex_shader(vs.main_entry_point(), ())
                    .fragment_shader(fs.main_entry_point(), ())
                    .build(Arc::clone(device))
                    .expect("Could not create graphics pipeline")
            )
        }
        (VertexBuf::UV(_), Material::GlTFPBRMaterial{base_color_texture, ..} ) => {
            if base_color_texture.is_some() {
                let vs = vs_pbr_base_color_texture::Shader::load(Arc::clone(device))
                    .expect("Vertex shader compilation failed");
                let fs = fs_pbr_base_color_texture::Shader::load(Arc::clone(device))
                    .expect("Fragment shader compilation failed");

                Arc::new(
                    builder
                    .vertex_input_single_buffer::<VertexUV>()
                    // How to interpret the vertex input
                    .vertex_shader(vs.main_entry_point(), ())
                    .fragment_shader(fs.main_entry_point(), ())
                    .build(Arc::clone(device))
                    .expect("Could not create graphics pipeline")
                )
            } else {
                unimplemented!()
            }
        }
        (VertexBuf::UVCol(_), Material::GlTFPBRMaterial{base_color_texture, ..} ) => {
            if base_color_texture.is_none() {
                let vs = vs_pbr_uv_col::Shader::load(Arc::clone(device))
                    .expect("Vertex shader compilation failed");
                let fs = fs_pbr_uv_col::Shader::load(Arc::clone(device))
                    .expect("Fragment shader compilation failed");

                Arc::new(
                    builder
                    .vertex_input_single_buffer::<VertexUVCol>()
                    // How to interpret the vertex input
                    .vertex_shader(vs.main_entry_point(), ())
                    .fragment_shader(fs.main_entry_point(), ())
                    .build(Arc::clone(device))
                    .expect("Could not create graphics pipeline")
                )
            } else {
                unimplemented!()
            }
        }
        (_, _) => unimplemented!()
    }
}
