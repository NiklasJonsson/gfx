use crate::ecs::prelude::*;
use crate::math::{
    orthographic_vk, perspective_vk, Aabb, FrustrumPlanes, Mat4, Obb, Quat, Rgb, Rgba, Transform,
    Vec2, Vec3, Vec4,
};
use crate::render::debug::LineConfig;
use crate::{imdbg, render};

use gltf::json::extensions::buffer::Buffer;
use trekant::{
    vk, BufferDescriptor, BufferHandle, BufferMutability, CommandBuffer, Extent2D,
    GraphicsPipeline, Handle, PipelineResourceSet, RenderPass, RenderTarget, Renderer,
    VertexFormat,
};

use crate::render::uniform::{LightingData, PackedLight, MAX_NUM_LIGHTS};
use std::ops::Range;

use super::imgui::UiFrame;
use super::shader::ShaderLocation;
use super::uniform::{self, DIRECTIONAL_SHADOW_MAP_COUNT};

#[derive(Component, ram_derive::Visitable, serde::Serialize, serde::Deserialize, Clone, Debug)]
pub enum Light {
    // Range is the radius of the sphere
    Point {
        color: Rgb,
        range: f32,
    },
    Directional {
        color: Rgb,
    },
    Spot {
        color: Rgb,
        /// The angle from the center line of the cone.
        angle: f32,
        /// `range` from the base to the point of the cone.
        range: Range<f32>,
    },
    Ambient {
        color: Rgb,
        strength: f32,
    },
}

impl Light {
    // As per gltf extension, KHR_light_punctual. Also makes sense as perspective matrix is based on
    // camera looking in -z
    pub const DEFAULT_FACING: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };
}

impl Default for Light {
    fn default() -> Self {
        Self::Spot {
            color: Rgb {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
            angle: std::f32::consts::FRAC_PI_8,
            range: Range {
                start: 0.1,
                end: 5.0,
            },
        }
    }
}

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct ShadowViewer;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShadowType {
    Directional = uniform::SHADOW_TYPE_DIRECTIONAL as isize,
    Point = uniform::SHADOW_TYPE_POINT as isize,
    Spot = uniform::SHADOW_TYPE_SPOT as isize,
}

#[derive(Clone, Copy, Component)]
pub struct ShadowMap {
    matrix_idx: u32,
    texture_idx: u32,
    shadow_type: ShadowType,
}

#[derive(Clone, Copy, Component)]
pub struct ShadowPipeline {
    directional: Handle<trekant::GraphicsPipeline>,
    spotlight: Handle<trekant::GraphicsPipeline>,
    pointlight: Handle<trekant::GraphicsPipeline>,
}

#[derive(Clone, Copy)]
struct ShadowRenderPassInfo {
    extent: Extent2D,
    render_target: Handle<RenderTarget>,
    view_data: Handle<PipelineResourceSet>,
    render_pass: Handle<trekant::RenderPass>,
    shadow_type: ShadowType,
}

/// Compute the bounds of the view are that we want to cast shadows on.
/// The coordinates are in world-space.
pub fn compute_world_shadow_bounds(world: &World) -> Option<Obb> {
    use crate::camera::Camera;

    type SysData<'a> = (
        ReadStorage<'a, Camera>,
        ReadStorage<'a, Transform>,
        ReadStorage<'a, ShadowViewer>,
    );

    let (cameras, transforms, markers) = SysData::fetch(world);
    let mut obb = None;
    for (cam, tfm, _marker) in (&cameras, &transforms, &markers).join() {
        let view_matrix = Mat4::from(*tfm).inverted();
        if obb.is_none() {
            obb = Some(view_matrix.inverted() * cam.view_obb());
        } else {
            log::error!("Too many shadow viewing cameras, ignoring all but first.");
        }
    }
    obb
}

fn compute_directional_shadow_bounds(shadow_bounds_ls: Obb, shadow_map_extent: Extent2D) -> Aabb {
    // Use the diagonal of the obb to define the size of the resulting aabb. This means
    // the aabb will have a fixed size (it's maximum) regardless of the orientation of the obb.
    let [u, v, w] = shadow_bounds_ls.uvw();
    let diagonal = Vec3::from(((u + v + w) * 2.0).magnitude());
    let mut aabb_lightspace = Aabb::from(shadow_bounds_ls);
    aabb_lightspace.max = aabb_lightspace.min + diagonal;

    // Ensure that the aabb only moves in texel-sized increments. This stops shadows from moving around as the
    // camera moves around (as the shadow aabb is computed from the camera). The idea is to fix the aabb to the
    // grid of the shadow map in space, ensuring that the rasterized objects during a shadow pass consistently
    // overlaps the right texel. Hard to explain in text but if a triangle is drawn across a grid. Then the grid
    // shifts very slightly (less than a texex size) so that the triangle no longer covers the centre of a cell (texel),
    // that depth buffer texel will no longer have the depth of the camera. If instead, the grid only shifts in texel-sized
    // increments. The triangle will always overlap the subtexel area in the same way, even if the grid has shifted.
    // See https://docs.microsoft.com/en-us{/windows/win32/dxtecharts/cascaded-shadow-maps for details.
    let shadow_map_extents = Vec2 {
        x: shadow_map_extent.width as f32,
        y: shadow_map_extent.height as f32,
    };

    let min = aabb_lightspace.min.xy();
    let max = aabb_lightspace.max.xy();

    let texel_size_lightspace = (max - min) / shadow_map_extents;

    let min = (min / texel_size_lightspace).floor() * texel_size_lightspace;
    let max = (max / texel_size_lightspace).floor() * texel_size_lightspace;

    Aabb {
        min: min.with_z(aabb_lightspace.min.z),
        max: max.with_z(aabb_lightspace.max.z),
    }
}

fn debug_shadow_bounds(world: &World, shadow_bounds_ws: Obb) {
    use std::fmt::Write as _;
    let debugger = world.read_resource::<super::debug::OneShotDebugUI>();
    debugger.add(move |_: &mut World, ui: &UiFrame| {
        let ui = ui.inner();
        let mut text = "Shadow bounds (light) (WS):\n".to_owned();
        let diag = shadow_bounds_ws.max_diagonal();

        writeln!(&mut text, "DIAG: {}, {}", diag[0], diag[1]).unwrap();

        for c in shadow_bounds_ws.corners() {
            write!(&mut text, "\n\t{},", c).unwrap();
        }
        text.truncate(text.len() - 1);
        ui.text(text);
    });

    let dr = world.read_resource::<super::debug::DebugRenderer>();
    dr.draw_obb(
        shadow_bounds_ws,
        LineConfig {
            color: Rgba::white(),
        },
    );
}

fn debug_directional_shadow_bounds(world: &World, light_bounds_ls: Obb, shadow_map_aabb: Aabb) {
    let debugger = world.read_resource::<super::debug::OneShotDebugUI>();
    debugger.add(move |_: &mut World, ui: &UiFrame| {
        let ui = ui.inner();
        let mut text = "Light space bounds (blue):".to_owned();
        for c in light_bounds_ls.corners() {
            text.push_str("\n\t");
            use std::fmt::Write as _;
            write!(&mut text, "{}", c).unwrap();
            text.push(',');
        }
        text.truncate(text.len() - 1);
        ui.text(text);

        ui.text(format!(
            "shadow_map_aabb (green): min {} {} {}. max {} {} {}",
            shadow_map_aabb.min.x,
            shadow_map_aabb.min.y,
            shadow_map_aabb.min.z,
            shadow_map_aabb.max.x,
            shadow_map_aabb.max.y,
            shadow_map_aabb.max.z
        ));
    });

    let dr = world.read_resource::<super::debug::DebugRenderer>();
    dr.draw_obb(
        light_bounds_ls,
        LineConfig {
            color: Rgba::blue(),
        },
    );
    dr.draw_aabb(
        shadow_map_aabb,
        LineConfig {
            color: Rgba::green(),
        },
    );
}

const N_SHADOW_PASSES_POINTLIGHT: usize = 6;

pub struct DirectionalShadow {
    pub render_target: Handle<RenderTarget>,
    pub light_info_buf: BufferHandle,
    pub light_info_prs: Handle<PipelineResourceSet>,
    pub texture: Handle<trekant::Texture>,
    pub extent: Extent2D,
}

#[derive(Clone, Copy)]
pub struct SpotlightShadow {
    pub render_target: Handle<RenderTarget>,
    pub light_info_buf: BufferHandle,
    pub light_info_prs: Handle<PipelineResourceSet>,
    pub texture: Handle<trekant::Texture>,
    pub extent: Extent2D,
}

#[derive(Clone, Copy)]
pub struct PointlightShadow {
    pub render_pass: Handle<trekant::RenderPass>,
    pub render_targets: [Handle<RenderTarget>; N_SHADOW_PASSES_POINTLIGHT],
    pub view_data_buffer: BufferHandle,
    pub view_data_pr_sets: [Handle<PipelineResourceSet>; N_SHADOW_PASSES_POINTLIGHT],
    // Cube
    pub cube_map: Handle<trekant::Texture>,
    pub depth_buffer: Handle<trekant::Texture>,
    pub extent: Extent2D,
}

pub struct ShadowResources {
    pub spotlight_render_pass: Handle<trekant::RenderPass>,
    pub directional_light_render_pass: Handle<trekant::RenderPass>,
    pub pointlight_render_pass: Handle<trekant::RenderPass>,
    pub shadow_light_info_buf: BufferHandle,
    pub directional: DirectionalShadow,
    pub spotlights: Vec<SpotlightShadow>,
    pub pointlights: Vec<PointlightShadow>,
    pub config: ShadowConfig,
    pub depth_dummy_pipeline: Handle<GraphicsPipeline>,
    pub pointlight_dummy_pipeline: Handle<GraphicsPipeline>,
}

pub fn prepare_entities(world: &World, renderer: &mut Renderer) {
    let shader_compiler = world.read_resource::<super::shader::ShaderCompiler>();
    let frame_data = world.read_resource::<super::FrameResources>();
    let meshes = world.read_storage::<super::Mesh>();
    let materials = world.read_storage::<super::GpuMaterial>();
    let mut renderables = world.write_storage::<ShadowPipeline>();
    let entities = world.entities();

    for (mesh, material, entity, _) in
        (&meshes, &materials, &entities, &!renderables.mask().clone()).join()
    {
        if let super::material::GpuMaterial::PBR { .. } = material {
            let vertex_format_size = mesh.cpu_vertex_buffer.format().size();
            let shadow_vertex_format = trekant::vertex::VertexFormat::builder()
                .add_attribute(trekant::util::Format::FLOAT3) // pos
                .skip(vertex_format_size - trekant::util::Format::FLOAT3.size())
                .build();

            // TODO: Cleanup error handling
            let directional_pipeline = {
                let desc =
                    depth_shadow_pipeline_desc(&shader_compiler, shadow_vertex_format.clone())
                        .unwrap();
                renderer
                    .create_gfx_pipeline(desc, &frame_data.shadow.directional_light_render_pass)
                    .unwrap()
            };

            let spotlight_pipeline = {
                let desc =
                    depth_shadow_pipeline_desc(&shader_compiler, shadow_vertex_format.clone())
                        .unwrap();
                renderer
                    .create_gfx_pipeline(desc, &frame_data.shadow.spotlight_render_pass)
                    .unwrap()
            };

            let pointlight_pipeline = {
                let desc =
                    pointlight_shadow_pipeline_desc(&shader_compiler, shadow_vertex_format.clone())
                        .unwrap();
                renderer
                    .create_gfx_pipeline(desc, &frame_data.shadow.pointlight_render_pass)
                    .unwrap()
            };

            if let Err(e) = renderables.insert(
                entity,
                ShadowPipeline {
                    spotlight: spotlight_pipeline,
                    directional: directional_pipeline,
                    pointlight: pointlight_pipeline,
                },
            ) {
                log::error!("Failed to insert shadow pipeline for {entity:?} due to {e}");
            }
        }
    }
}

fn create_shadow_render_pass_depth(
    renderer: &mut Renderer,
    config: &ShadowConfig,
) -> Handle<trekant::RenderPass> {
    let depth_attach = vk::AttachmentDescription {
        format: config.depth_texture_format.into(),
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        flags: vk::AttachmentDescriptionFlags::empty(),
    };

    let depth_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref);

    let deps = [
        vk::SubpassDependency {
            // The source pass deps here refer to any commands before this render pass in "submission order".
            src_subpass: vk::SUBPASS_EXTERNAL,
            // srevious fragment shader reads should be done
            src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_subpass: 0,
            // EARLY_FRAGMENT_TESTS include subpass load operations for depth/stencil
            dst_stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            // We are writing to the depth attachment
            dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            // We don't need a global dependency for the whole framebuffer
            dependency_flags: vk::DependencyFlags::BY_REGION,
        },
        vk::SubpassDependency {
            src_subpass: 0,
            // LATE_FRAGMENT_TESTS include subpass store operations for depth/stencil
            src_stage_mask: vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,

            // We want this render pass to complete before any subsequent uses of the depth buffer as a texture
            dst_subpass: vk::SUBPASS_EXTERNAL,
            dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            // Do we actually need a full dependency region here?
            dependency_flags: vk::DependencyFlags::empty(),
        },
    ];

    let attachments = [depth_attach];
    let subpasses = [subpass.build()];
    let create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&deps);

    renderer
        .create_render_pass(&create_info)
        .expect("Failed to create shadow render pass")
}

fn create_shadow_render_pass_pointlight(
    renderer: &mut Renderer,
    config: &ShadowConfig,
) -> Handle<trekant::RenderPass> {
    let color_buffer = vk::AttachmentDescription {
        format: config.pointlight.color_texture_format.into(),
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        flags: vk::AttachmentDescriptionFlags::empty(),
    };

    let depth_buffer = vk::AttachmentDescription {
        format: config.depth_texture_format.into(),
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        flags: vk::AttachmentDescriptionFlags::empty(),
    };

    let color_attach = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let depth_attach = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attach)
        .depth_stencil_attachment(&depth_attach);

    let deps = [
        vk::SubpassDependency {
            // The source pass deps here refer to any commands before this render pass in "submission order".
            src_subpass: vk::SUBPASS_EXTERNAL,
            // Any previous fragment shader reads should be done
            src_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access_mask: vk::AccessFlags::SHADER_READ,
            dst_subpass: 0,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        },
        vk::SubpassDependency {
            src_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            // We want this render pass to complete before any subsequent uses of the texture
            dst_subpass: vk::SUBPASS_EXTERNAL,
            dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            dependency_flags: vk::DependencyFlags::empty(),
        },
    ];

    let attachments = [color_buffer, depth_buffer];
    let subpasses = [subpass.build()];
    let create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&deps);

    renderer
        .create_render_pass(&create_info)
        .expect("Failed to create shadow render pass")
}

fn depth_shadow_pipeline_desc(
    shader_compiler: &super::shader::ShaderCompiler,
    format: VertexFormat,
) -> Result<trekant::GraphicsPipelineDescriptor, super::MaterialError> {
    let no_defines = super::shader::Defines::empty();
    let vert = shader_compiler.compile(
        &ShaderLocation::builtin("render/shaders/shadow/depth_write_vert.glsl"),
        &no_defines,
        super::shader::ShaderType::Vertex,
    )?;

    let vert = super::ShaderDescriptor {
        debug_name: Some("shadow-depth-write-vert".to_owned()),
        spirv_code: vert.data(),
    };

    Ok(trekant::GraphicsPipelineDescriptor::builder()
        .vertex_format(format)
        .vert(vert)
        .culling(trekant::pipeline::TriangleCulling::Front)
        .build()?)
}

fn pointlight_shadow_pipeline_desc(
    shader_compiler: &super::shader::ShaderCompiler,
    format: VertexFormat,
) -> Result<trekant::GraphicsPipelineDescriptor, super::MaterialError> {
    let no_defines = super::shader::Defines::empty();
    let vert = {
        let vert = shader_compiler.compile(
            &ShaderLocation::builtin("render/shaders/shadow/pointlight_vert.glsl"),
            &no_defines,
            super::shader::ShaderType::Vertex,
        )?;

        super::ShaderDescriptor {
            debug_name: Some("pointlight-shadow-vert".to_owned()),
            spirv_code: vert.data(),
        }
    };
    let frag = {
        let frag = shader_compiler.compile(
            &ShaderLocation::builtin("render/shaders/shadow/pointlight_frag.glsl"),
            &no_defines,
            super::shader::ShaderType::Fragment,
        )?;

        super::ShaderDescriptor {
            debug_name: Some("pointlight-shadow-frag".to_owned()),
            spirv_code: frag.data(),
        }
    };

    Ok(trekant::GraphicsPipelineDescriptor::builder()
        .vertex_format(format)
        .vert(vert)
        .frag(frag)
        .culling(trekant::pipeline::TriangleCulling::Front)
        .build()?)
}

#[derive(Debug, Clone, Copy)]
struct PointlightShadowConfig {
    color_texture_format: trekant::Format,
    depth_texture_format: trekant::Format,
    max: u32,
    shadow_map_extent: Extent2D,
}

pub struct ShadowConfig {
    spotlight_shadow_map_extent: Extent2D,
    directional_light_shadow_map_extent: Extent2D,
    depth_texture_format: trekant::Format,
    max_directional_lights: u32,
    max_spotlights: u32,
    pointlight: PointlightShadowConfig,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        ShadowConfig {
            spotlight_shadow_map_extent: Extent2D {
                width: 1024,
                height: 1024,
            },
            directional_light_shadow_map_extent: Extent2D {
                width: 4096,
                height: 4096,
            },
            pointlight: PointlightShadowConfig {
                shadow_map_extent: Extent2D {
                    width: 512,
                    height: 512,
                },
                max: uniform::POINTLIGHT_SHADOW_MAP_COUNT,
                color_texture_format: trekant::Format::FLOAT1,
                depth_texture_format: trekant::Format::D16_UNORM,
            },
            max_directional_lights: uniform::DIRECTIONAL_SHADOW_MAP_COUNT,
            max_spotlights: uniform::SPOTLIGHT_SHADOW_MAP_COUNT,
            depth_texture_format: trekant::Format::D16_UNORM,
        }
    }
}

impl ShadowConfig {
    fn max_num_views(&self) -> u32 {
        self.max_directional_lights + self.max_spotlights + self.pointlight.max * 6
    }
}

pub fn setup_shadow_resources(
    shader_compiler: &super::shader::ShaderCompiler,
    renderer: &mut Renderer,
) -> ShadowResources {
    use trekant::{BorderColor, Filter, SamplerAddressMode};
    let config = ShadowConfig::default();

    let pointlight_render_pass = create_shadow_render_pass_pointlight(renderer, &config);
    let depth_render_pass = create_shadow_render_pass_depth(renderer, &config);

    let shadow_light_info_buf = {
        let initial_data =
            vec![uniform::ShadowLightInfo::default(); config.max_num_views() as usize];
        let desc = BufferDescriptor::uniform_buffer(
            &initial_data,
            BufferMutability::Mutable,
            trekant::BufferLayout::MinBufferOffset,
        );

        renderer
            .create_buffer(desc)
            .expect("Failed to create buffer for shadow view and projection matrices")
    };

    let mut next_info_buf = shadow_light_info_buf;
    let directional = {
        let extent = config.directional_light_shadow_map_extent;
        let view_data = next_info_buf.take_first(1);
        let (texture, render_target) = {
            let desc = trekant::TextureDescriptor::Empty {
                extent,
                format: config.depth_texture_format,
                usage: trekant::TextureUsage::Depth,
                sampler: trekant::SamplerDescriptor {
                    filter: Filter::Linear,
                    address_mode: SamplerAddressMode::ClampToEdge,
                    max_anisotropy: None,
                    border_color: BorderColor::FloatOpaqueWhite,
                },
                ty: trekant::TextureType::Tex2D,
            };
            let texture = renderer
                .create_texture(desc)
                .expect("Failed to create texture for shadow map");
            let attachments = [trekant::RenderTargetAttachment {
                texture,
                range: trekant::TextureImageRange::Full,
            }];
            let render_target = renderer
                .create_render_target(depth_render_pass, &attachments)
                .expect("Failed to create render target for shadow map");
            (texture, render_target)
        };
        let sh_view_data_set = PipelineResourceSet::builder(renderer)
            .add_buffer(view_data, 0, trekant::ShaderStage::VERTEX)
            .build();

        DirectionalShadow {
            texture,
            render_target,
            light_info_buf: view_data,
            light_info_prs: sh_view_data_set,
            extent,
        }
    };

    let mut spotlights = Vec::with_capacity(config.max_spotlights as usize);
    for _ in 0..config.max_spotlights {
        let spotlight = {
            let extent = config.spotlight_shadow_map_extent;
            let light_info = next_info_buf.take_first(1);
            let (texture, render_target) = {
                let desc = trekant::TextureDescriptor::Empty {
                    extent,
                    format: config.depth_texture_format,
                    usage: trekant::TextureUsage::Depth,
                    sampler: trekant::SamplerDescriptor {
                        filter: Filter::Linear,
                        address_mode: SamplerAddressMode::ClampToEdge,
                        max_anisotropy: None,
                        border_color: BorderColor::FloatOpaqueWhite,
                    },
                    ty: trekant::TextureType::Tex2D,
                };
                let texture = renderer
                    .create_texture(desc)
                    .expect("Failed to create texture for shadow map");
                let attachments = [trekant::RenderTargetAttachment {
                    texture,
                    range: trekant::TextureImageRange::Full,
                }];
                let render_target = renderer
                    .create_render_target(depth_render_pass, &attachments)
                    .expect("Failed to create render target for shadow map");
                (texture, render_target)
            };
            let sh_view_data_set = PipelineResourceSet::builder(renderer)
                .add_buffer(light_info, 0, trekant::ShaderStage::VERTEX)
                .build();

            SpotlightShadow {
                texture,
                render_target,
                light_info_buf: light_info,
                light_info_prs: sh_view_data_set,
                extent,
            }
        };
        spotlights.push(spotlight);
    }

    let mut pointlights = Vec::with_capacity(config.pointlight.max as usize);
    for _ in 0..config.pointlight.max {
        let extent = config.pointlight.shadow_map_extent;
        let cube_desc = trekant::TextureDescriptor::Empty {
            extent,
            format: config.pointlight.color_texture_format,
            usage: trekant::TextureUsage::Color,
            sampler: trekant::SamplerDescriptor {
                filter: trekant::Filter::Linear,
                address_mode: trekant::SamplerAddressMode::ClampToEdge,
                max_anisotropy: None,
                border_color: trekant::BorderColor::FloatOpaqueWhite,
            },
            ty: trekant::TextureType::TexCube,
        };

        let cube_map = renderer
            .create_texture(cube_desc)
            .expect("Failed to create texture for shadow map");

        let depth_desc = trekant::TextureDescriptor::Empty {
            extent,
            format: config.pointlight.depth_texture_format,
            usage: trekant::TextureUsage::Depth,
            sampler: trekant::SamplerDescriptor {
                filter: trekant::Filter::Linear,
                address_mode: trekant::SamplerAddressMode::ClampToEdge,
                max_anisotropy: None,
                border_color: trekant::BorderColor::FloatOpaqueWhite,
            },
            ty: trekant::TextureType::Tex2D,
        };

        let depth_buffer = renderer
            .create_texture(depth_desc)
            .expect("Failed to create texture for shadow map");

        let render_targets = {
            let render_targets: [Handle<trekant::RenderTarget>; 6] = std::array::from_fn(|idx| {
                let attachments = [
                    trekant::RenderTargetAttachment {
                        texture: cube_map,
                        range: trekant::TextureImageRange::Part { start: idx, len: 1 },
                    },
                    trekant::RenderTargetAttachment {
                        texture: depth_buffer,
                        range: trekant::TextureImageRange::Full,
                    },
                ];
                renderer
                    .create_render_target(pointlight_render_pass, &attachments)
                    .expect("Failed to create render target for cubemap")
            });
            render_targets
        };

        let light_info_buffer = next_info_buf.take_first(6);
        let pr_sets = std::array::from_fn(|i| {
            PipelineResourceSet::builder(renderer)
                .add_buffer(
                    light_info_buffer.slice(i as u32, 1),
                    0,
                    trekant::ShaderStage::VERTEX | trekant::ShaderStage::FRAGMENT,
                )
                .build()
        });
        pointlights.push(PointlightShadow {
            render_pass: pointlight_render_pass,
            render_targets,
            view_data_buffer: light_info_buffer,
            view_data_pr_sets: pr_sets,
            cube_map,
            extent,
            depth_buffer,
        })
    }

    let depth_dummy_pipeline = {
        let pos_only_vertex_format = VertexFormat::from(trekant::Format::FLOAT3);
        let pipeline_desc = depth_shadow_pipeline_desc(shader_compiler, pos_only_vertex_format)
            .expect("failed to create graphics pipeline descriptor for shadows");
        renderer
            .create_gfx_pipeline(pipeline_desc, &depth_render_pass)
            .expect("failed to create pipeline for shadow")
    };

    let pointlight_dummy_pipeline = {
        let pos_only_vertex_format = VertexFormat::from(trekant::Format::FLOAT3);
        let pipeline_desc =
            pointlight_shadow_pipeline_desc(shader_compiler, pos_only_vertex_format)
                .expect("failed to create graphics pipeline descriptor for shadows");
        renderer
            .create_gfx_pipeline(pipeline_desc, &pointlight_render_pass)
            .expect("failed to create pipeline for shadow")
    };

    ShadowResources {
        spotlight_render_pass: depth_render_pass,
        directional_light_render_pass: depth_render_pass,
        pointlight_render_pass,
        depth_dummy_pipeline,
        pointlight_dummy_pipeline,
        spotlights,
        shadow_light_info_buf,
        directional,
        pointlights,
        config,
    }
}

pub struct ShadowPassOutput {
    pub shadow_matrices: Vec<uniform::Mat4>,
}

fn transition_unused_textures(
    cmdbuf: &mut CommandBuffer,
    frame: &mut trekant::Frame,
    depth_textures: &[Handle<trekant::Texture>],
    cubemap_textures: &[Handle<trekant::Texture>],
) {
    let mut barriers = Vec::with_capacity(depth_textures.len() + cubemap_textures.len());
    for texture in depth_textures {
        let texture = frame
            .get_texture(texture)
            .expect("Failed to get shadow texture for mem barrier");
        let vk_image = texture.vk_image();

        barriers.push(vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: vk_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..Default::default()
        })
    }

    for texture in cubemap_textures {
        let texture = frame
            .get_texture(texture)
            .expect("Failed to get shadow texture for mem barrier");
        let vk_image = texture.vk_image();

        barriers.push(vk::ImageMemoryBarrier {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: vk_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            },
            src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            ..Default::default()
        })
    }

    cmdbuf.pipeline_barrier(
        &barriers,
        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    );
}

const POINTLIGHT_DIRECTIONS: [Vec3; N_SHADOW_PASSES_POINTLIGHT] = [
    Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    },
    Vec3 {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    },
    Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    },
    Vec3 {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    },
    Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    },
    Vec3 {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    },
];

struct ShadowRenderPasses {
    // The matrices for each of the shadow passes
    shadow_light_info: Vec<uniform::ShadowLightInfo>,
    render_passes: Vec<ShadowRenderPassInfo>,
    shadow_maps: Vec<(Entity, ShadowMap)>,
    shadow_pass_output: ShadowPassOutput,
    unused_depth_textures: Vec<Handle<trekant::Texture>>,
    unused_pointlight_textures: Vec<Handle<trekant::Texture>>,
}

impl ShadowRenderPasses {
    fn new(resources: &ShadowResources) -> Self {
        assert!(resources.shadow_light_info_buf.len() >= resources.config.max_num_views());
        let cap = resources.config.max_num_views() as usize;

        // TODO(perf): Alloc-reuse
        Self {
            shadow_light_info: vec![uniform::ShadowLightInfo::default(); cap],
            render_passes: Vec::with_capacity(cap),
            shadow_maps: Vec::with_capacity(cap),
            shadow_pass_output: ShadowPassOutput {
                // NOTE: This vector is technically over-allocated as there are 6 render passes for one point light
                shadow_matrices: Vec::with_capacity(cap),
            },
            unused_depth_textures: Vec::with_capacity(cap),
            unused_pointlight_textures: Vec::with_capacity(cap),
        }
    }

    // For directional and spotlights, the matrix is the combined view-projection matrix
    // but for point lights, this is just the view matrix as we don't need projection
    // for the cubemap.
    // We could also consider making this function take more granular information, e.g. the light position,
    // so that the main render pass can construct its own matrices. This would reduce the implicit coupling
    // a bit but would complicate the code in the main render pass.
    fn add_shadow_map(
        &mut self,
        entity: Entity,
        mtx: Mat4,
        texture_idx: u32,
        shadow_type: ShadowType,
    ) {
        let matrix_idx = self
            .shadow_pass_output
            .shadow_matrices
            .len()
            .try_into()
            .unwrap();
        self.shadow_pass_output
            .shadow_matrices
            .push(mtx.into_col_array());
        self.shadow_maps.push((
            entity,
            ShadowMap {
                matrix_idx,
                texture_idx,
                shadow_type,
            },
        ));
    }

    #[allow(clippy::too_many_arguments)]
    fn add_shadow_pass(
        &mut self,
        render_pass: Handle<trekant::RenderPass>,
        render_target: Handle<RenderTarget>,
        view_data_buffer: BufferHandle,
        view_data_desc_set: Handle<PipelineResourceSet>,
        extent: Extent2D,
        viewproj: Mat4,
        pos: Vec3,
        shadow_type: ShadowType,
    ) {
        self.render_passes.push(ShadowRenderPassInfo {
            render_pass,
            extent,
            render_target,
            view_data: view_data_desc_set,
            shadow_type,
        });
        let matrix_buf_idx = view_data_buffer.offset() as usize;
        self.shadow_light_info[matrix_buf_idx] = uniform::ShadowLightInfo {
            view_proj: viewproj.into_col_array(),
            pos: pos.with_w(1.0).into_array(),
        };
    }

    fn n_lights(&self) -> usize {
        self.shadow_pass_output.shadow_matrices.len()
    }
}

fn collect_shadow_passes(world: &World, shadow_resources: &ShadowResources) -> ShadowRenderPasses {
    let shadow_bounds_ws =
        compute_world_shadow_bounds(world).expect("Failed to compute bounds for shadows");

    {
        debug_shadow_bounds(world, shadow_bounds_ws);
    }

    // Always make room for the directional light
    const MAX_NUM_DIRECTIONAL_LIGHTS: u32 = 1;
    const MAX_NUM_NON_DIRECTIONAL_LIGHTS: u32 = MAX_NUM_LIGHTS as u32 - MAX_NUM_DIRECTIONAL_LIGHTS;
    let mut n_spotlights: u32 = 0;
    let mut n_pointlights: u32 = 0;
    let mut n_directional_lights: u32 = 0;

    let mut shadow_render_passes = ShadowRenderPasses::new(shadow_resources);

    let lights = world.read_storage::<Light>();
    let transforms = world.read_storage::<Transform>();
    let entities = world.read_resource::<EntitiesRes>();
    for (light_entity, light, tfm) in (&entities, &lights, &transforms).join() {
        if shadow_render_passes.n_lights() >= MAX_NUM_LIGHTS {
            log::warn!("Too many lights, skipping remaining");
            break;
        }

        let n_non_directional_lights = n_pointlights + n_spotlights;

        let to_lightspace = Mat4::from(*tfm).inverted();
        match light {
            Light::Spot { angle, range, .. } => {
                if n_non_directional_lights > MAX_NUM_NON_DIRECTIONAL_LIGHTS {
                    log::warn!(
                        "Too many spot lights, skipping shadow generation for {light_entity:?}"
                    );
                    continue;
                }
                // TODO: Aspect ratio based on texture?
                let mtx = perspective_vk(angle * 2.0, 1.0, range.start, range.end) * to_lightspace;

                let spotlight_idx = n_spotlights;
                n_spotlights += 1;

                let resource = shadow_resources.spotlights[spotlight_idx as usize];

                shadow_render_passes.add_shadow_map(
                    light_entity,
                    mtx,
                    spotlight_idx,
                    ShadowType::Spot,
                );
                shadow_render_passes.add_shadow_pass(
                    shadow_resources.spotlight_render_pass,
                    resource.render_target,
                    resource.light_info_buf,
                    resource.light_info_prs,
                    resource.extent,
                    mtx,
                    tfm.position,
                    ShadowType::Spot,
                );
            }
            Light::Directional { .. } => {
                if n_directional_lights >= MAX_NUM_DIRECTIONAL_LIGHTS {
                    log::warn!("Found more than {MAX_NUM_DIRECTIONAL_LIGHTS} directional lights, skipping {light_entity:?}");
                    continue;
                }

                let directional_light_idx = 0;
                n_directional_lights += 1;

                let resource = &shadow_resources.directional;

                let light_bounds_ls = to_lightspace * shadow_bounds_ws;
                let aabb = compute_directional_shadow_bounds(light_bounds_ls, resource.extent);
                {
                    debug_directional_shadow_bounds(world, light_bounds_ls, aabb);
                }

                let proj: Mat4 = orthographic_vk(FrustrumPlanes {
                    left: aabb.min.x,
                    right: aabb.max.x,
                    bottom: aabb.min.y,
                    top: aabb.max.y,
                    near: aabb.min.z,
                    far: aabb.max.z,
                });

                let mtx = proj * to_lightspace;
                shadow_render_passes.add_shadow_map(
                    light_entity,
                    mtx,
                    directional_light_idx,
                    ShadowType::Directional,
                );
                shadow_render_passes.add_shadow_pass(
                    shadow_resources.directional_light_render_pass,
                    resource.render_target,
                    resource.light_info_buf,
                    resource.light_info_prs,
                    resource.extent,
                    mtx,
                    tfm.position,
                    ShadowType::Directional,
                );
            }
            Light::Point { range, .. } => {
                if n_non_directional_lights > MAX_NUM_NON_DIRECTIONAL_LIGHTS {
                    log::warn!(
                        "Too many point lights, skipping shadow generation for {light_entity:?}"
                    );
                    break;
                }

                let pointlight_idx = n_pointlights;
                n_pointlights += 1;

                let pointlight_shadow = shadow_resources.pointlights[pointlight_idx as usize];

                const NEAR: f32 = 1.0;
                let aspect_ratio: f32 =
                    pointlight_shadow.extent.width as f32 / pointlight_shadow.extent.height as f32;
                let far = *range;
                let perspective =
                    perspective_vk(std::f32::consts::FRAC_PI_2, aspect_ratio, NEAR, far);
                shadow_render_passes.add_shadow_map(
                    light_entity,
                    // NOTE: No projection matrix here as the cube map is sampled with a direction vector in light-space
                    to_lightspace,
                    pointlight_idx,
                    ShadowType::Point,
                );

                for (pass_idx, cube_face_dir) in POINTLIGHT_DIRECTIONS.into_iter().enumerate() {
                    let shadow_view = Mat4::from(Transform {
                        position: tfm.position,
                        rotation: Quat::rotation_from_to_3d(Light::DEFAULT_FACING, cube_face_dir),
                        scale: 1.0,
                    })
                    .inverted();

                    let viewproj = perspective * shadow_view;
                    let matrix_buffer_elem: BufferHandle =
                        pointlight_shadow.view_data_buffer.slice(pass_idx as u32, 1);

                    shadow_render_passes.add_shadow_pass(
                        shadow_resources.pointlight_render_pass,
                        pointlight_shadow.render_targets[pass_idx],
                        matrix_buffer_elem,
                        pointlight_shadow.view_data_pr_sets[pass_idx],
                        pointlight_shadow.extent,
                        viewproj,
                        tfm.position,
                        ShadowType::Point,
                    );
                }
            }
            Light::Ambient { .. } => continue,
        };
    }

    let unused_itr = shadow_resources
        .spotlights
        .iter()
        .skip(n_spotlights as usize)
        .map(|l| l.texture);
    let unused_itr = unused_itr.chain(
        std::iter::once(shadow_resources.directional.texture).skip(n_directional_lights as usize),
    );
    shadow_render_passes.unused_depth_textures = unused_itr.collect();

    shadow_render_passes.unused_pointlight_textures = shadow_resources
        .pointlights
        .iter()
        .skip(n_pointlights as usize)
        .map(|l| l.cube_map)
        .collect();

    shadow_render_passes
}

#[profiling::function]
pub fn draw_entities_shadow(
    world: &World,
    cmd_buf: &mut trekant::RenderPassEncoder<'_>,
    shadow_type: ShadowType,
) {
    let model_matrices = world.read_storage::<super::ModelMatrix>();
    let meshes = world.read_storage::<super::Mesh>();
    let pipelines = world.read_storage::<ShadowPipeline>();
    use trekant::pipeline::ShaderStage;

    let mut prev_handle: Option<Handle<GraphicsPipeline>> = None;

    for (mesh, pipeline, mtx) in (&meshes, &pipelines, &model_matrices).join() {
        let (super::GpuResource::Available(vbuf), super::GpuResource::Available(ibuf)) =
            (&mesh.gpu_vertex_buffer, &mesh.gpu_index_buffer)
        else {
            continue;
        };

        let tfm = uniform::Model {
            model: mtx.0.into_col_array(),
            model_it: mtx.0.inverted().transposed().into_col_array(),
        };

        let pipeline_handle = match shadow_type {
            ShadowType::Directional => pipeline.directional,
            ShadowType::Point => pipeline.pointlight,
            ShadowType::Spot => pipeline.spotlight,
        };

        let do_bind = prev_handle.map(|h| h != pipeline_handle).unwrap_or(true);
        if do_bind {
            cmd_buf.bind_graphics_pipeline(&pipeline_handle);
            prev_handle = Some(pipeline_handle);
        }
        cmd_buf
            .bind_push_constant(&pipeline_handle, ShaderStage::VERTEX, &tfm)
            .draw_mesh(*vbuf, *ibuf);
    }
}

// TODO: Move to config
const POINTLIGHT_CLEAR_VALUES: &[vk::ClearValue] = &[
    vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [f32::MAX; 4],
        },
    },
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    },
];

const DEPTH_ONLY_CLEAR_VALUES: &[vk::ClearValue] = &[vk::ClearValue {
    depth_stencil: vk::ClearDepthStencilValue {
        depth: 1.0,
        stencil: 0,
    },
}];

pub fn shadow_pass(
    world: &World,
    frame: &mut trekant::Frame,
    shadow_resources: &ShadowResources,
    mut cmd_buffer: CommandBuffer,
) -> (CommandBuffer, ShadowPassOutput) {
    let shadow_render_passes = collect_shadow_passes(world, shadow_resources);
    frame
        .write_buffer(
            shadow_resources.shadow_light_info_buf,
            &shadow_render_passes.shadow_light_info,
        )
        .expect("Failed to update matrices for shadow coords");

    let mut shadow_maps = world.write_storage::<ShadowMap>();

    // Render passes
    for render_pass in shadow_render_passes.render_passes {
        let ShadowRenderPassInfo {
            extent,
            render_target,
            view_data,
            render_pass,
            shadow_type,
        } = render_pass;

        let (clear_values, dummy_pipeline) = match shadow_type {
            ShadowType::Spot => (
                DEPTH_ONLY_CLEAR_VALUES,
                shadow_resources.depth_dummy_pipeline,
            ),
            ShadowType::Directional => (
                DEPTH_ONLY_CLEAR_VALUES,
                shadow_resources.depth_dummy_pipeline,
            ),
            ShadowType::Point => (
                POINTLIGHT_CLEAR_VALUES,
                shadow_resources.pointlight_dummy_pipeline,
            ),
        };

        let mut shadow_rp = frame
            .begin_render_pass(
                cmd_buffer,
                &render_pass,
                &render_target,
                extent,
                clear_values,
            )
            .expect("Failed to shadow begin render pass");

        shadow_rp
            .bind_graphics_pipeline(&dummy_pipeline)
            .bind_shader_resource_group(0u32, &view_data, &dummy_pipeline);
        draw_entities_shadow(world, &mut shadow_rp, shadow_type);
        cmd_buffer = shadow_rp.end().expect("Failed to end shadow render pass");
    }

    // TODO: Get rid of the intermediate entities
    for (entity, shadow_map) in shadow_render_passes.shadow_maps {
        shadow_maps
            .insert(entity, shadow_map)
            .expect("Failed to add shadow map for light entity");
    }

    transition_unused_textures(
        &mut cmd_buffer,
        frame,
        &shadow_render_passes.unused_depth_textures,
        &shadow_render_passes.unused_pointlight_textures,
    );

    (cmd_buffer, shadow_render_passes.shadow_pass_output)
}

pub fn write_lighting_data(
    world: &World,
    frame: &mut trekant::Frame,
    frame_resources: &super::FrameResources,
    shadow_pass_output: &ShadowPassOutput,
) {
    let mut lighting_data = LightingData::default();

    let lights = world.read_storage::<Light>();
    let transforms = world.read_storage::<Transform>();
    let shadow_maps = world.read_storage::<ShadowMap>();
    let mut n_ambients = 0;
    let mut packed_light_count = 0;

    for (light, tfm, shadow_map) in (&lights, &transforms, (&shadow_maps).maybe()).join() {
        if let Light::Ambient { color, strength } = &light {
            if n_ambients > 0 {
                log::warn!("Too many ambient lights, skipping all but first");
            } else {
                lighting_data.ambient = [color.r, color.g, color.b, *strength];
                n_ambients = 1;
            }
            continue;
        }

        if packed_light_count as usize >= lighting_data.lights.len() {
            log::warn!("Too many punctual lights, skipping remaining");
            break;
        }

        let direction = tfm.rotation * Light::DEFAULT_FACING;
        let packed_light = match light {
            Light::Spot {
                angle,
                color,
                range,
            } => {
                let pos = tfm.position.with_w(1.0).into_array();

                PackedLight {
                    pos,
                    dir_cutoff: [direction.x, direction.y, direction.z, angle.cos()],
                    color_range: [color.r, color.g, color.b, range.end],
                    ..Default::default()
                }
            }
            Light::Directional { color } => PackedLight {
                pos: [0.0, 0.0, 0.0, 0.0],
                dir_cutoff: [direction.x, direction.y, direction.z, 0.0],
                color_range: [color.r, color.g, color.b, 0.0],
                ..Default::default()
            },
            Light::Point { color, range } => PackedLight {
                pos: tfm.position.with_w(1.0).into_array(),
                dir_cutoff: [0.0, 0.0, 0.0, 0.0],
                color_range: [color.r, color.g, color.b, *range],
                ..Default::default()
            },
            Light::Ambient { .. } => unreachable!("Should have been handled already"),
        };
        let light = &mut lighting_data.lights[packed_light_count as usize];
        *light = packed_light;

        packed_light_count += 1;
        if let Some(ShadowMap {
            matrix_idx,
            shadow_type,
            texture_idx,
        }) = shadow_map
        {
            light.shadow_info = [*shadow_type as u32, *matrix_idx, *texture_idx, u32::MAX];
        }
    }
    lighting_data.num_lights = [packed_light_count; 4];

    frame
        .write_buffer_element(
            frame_resources.engine_shader_resources.lighting_data,
            &lighting_data,
            0,
        )
        .expect("Failed to update uniform for lighting data");

    frame
        .write_buffer(
            frame_resources.engine_shader_resources.world_to_shadow,
            &shadow_pass_output.shadow_matrices,
        )
        .expect("Failed to write the shadow matrices storage buffer for the main render pass");
}
