use crate::ecs::prelude::*;
use crate::math::{
    orthographic_vk, perspective_vk, Aabb, FrustrumPlanes, Mat4, Obb, Rgb, Rgba, Transform, Vec2,
    Vec3,
};
use crate::render::debug::LineConfig;

use trekant::{
    vk, BufferHandle, BufferMutability, CommandBuffer, DescriptorSet, DeviceUniformBuffer,
    Extent2D, GraphicsPipeline, Handle, RenderPass, RenderTarget, Renderer, ResourceManager,
    UniformBufferDescriptor, VertexFormat,
};

use crate::render::uniform::{LightingData, PackedLight, MAX_NUM_LIGHTS};
use std::ops::Range;

use super::imgui::UiFrame;
use super::shader::ShaderLocation;
use super::uniform;

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

#[derive(Clone, Copy)]
struct ShadowRenderPassInfo {
    extent: Extent2D,
    render_target: Handle<RenderTarget>,
    view_data_buf: BufferHandle<DeviceUniformBuffer>,
    view_data: Handle<DescriptorSet>,
    shadow_map: ShadowMap,
    light_entity: Entity,
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
        // TODO: Some sites suggest to use view_proj.inverse() * NDC cube, try this.
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

pub struct Shadow {
    pub render_target: Handle<RenderTarget>,
    pub view_data_buffer: BufferHandle<DeviceUniformBuffer>,
    pub view_data_desc_set: Handle<DescriptorSet>,
    pub texture: Handle<trekant::Texture>,
    pub extent: Extent2D,
}

pub struct ShadowResources {
    pub render_pass: Handle<trekant::RenderPass>,
    pub dummy_pipeline: Handle<GraphicsPipeline>,
    pub shadow_matrices_buf: BufferHandle<DeviceUniformBuffer>,
    pub directional: Shadow,
    pub spotlights: Vec<Shadow>,
}

fn shadow_render_target(
    renderer: &mut Renderer,
    render_pass: &Handle<trekant::RenderPass>,
    extent: Extent2D,
) -> (Handle<trekant::Texture>, Handle<trekant::RenderTarget>) {
    use trekant::texture::{BorderColor, Filter, SamplerAddressMode};
    let format = trekant::Format::D16_UNORM;

    let desc = trekant::TextureDescriptor::Empty {
        extent,
        format,
        usage: trekant::TextureUsage::DEPTH_STENCIL_ATTACHMENT,
        sampler: trekant::SamplerDescriptor {
            filter: Filter::Linear,
            address_mode: SamplerAddressMode::ClampToEdge,
            max_anisotropy: None,
            border_color: BorderColor::FloatOpaqueWhite,
        },
    };
    let tex = renderer
        .create_texture(desc)
        .expect("Failed to create texture for shadow map");
    let attachments = [&tex];
    let render_target = renderer
        .create_render_target(render_pass, &attachments)
        .expect("Failed to create render target for shadow map");
    (tex, render_target)
}

fn create_shadow_render_pass(renderer: &mut Renderer) -> Handle<trekant::RenderPass> {
    let depth_attach = vk::AttachmentDescription {
        format: vk::Format::D16_UNORM,
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

    // These subpass dependencies handle layout transistions, execution & memory dependencies
    // When there are multiple shadow passes, it might be valuable to use one pipeline barrier
    // for all of them instead of several subpass deps.
    let deps = [
        vk::SubpassDependency {
            // The source pass deps here refer to the previous frame (I think :))
            src_subpass: vk::SUBPASS_EXTERNAL,
            // Any previous fragment shader reads should be done
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

fn shadow_pipeline_desc(
    shader_compiler: &super::shader::ShaderCompiler,
    format: VertexFormat,
) -> Result<trekant::GraphicsPipelineDescriptor, super::MaterialError> {
    let no_defines = super::shader::Defines::empty();
    let vert = shader_compiler.compile(
        &ShaderLocation::builtin("render/shaders/pos_only_vert.glsl"),
        &no_defines,
        super::shader::ShaderType::Vertex,
    )?;

    let vert = super::ShaderDescriptor {
        debug_name: Some("shadow-pos-only-vert".to_owned()),
        spirv_code: vert.data(),
    };

    Ok(trekant::GraphicsPipelineDescriptor::builder()
        .vertex_format(format)
        .vert(vert)
        .culling(trekant::pipeline::TriangleCulling::Front)
        .build()?)
}

fn build_single_shadow(
    renderer: &mut Renderer,
    view_data: BufferHandle<DeviceUniformBuffer>,
    shadow_render_pass: Handle<RenderPass>,
    extent: Extent2D,
) -> Shadow {
    let (texture, render_target) = shadow_render_target(renderer, &shadow_render_pass, extent);
    let sh_view_data_set = DescriptorSet::builder(renderer)
        .add_buffer(&view_data, 0, trekant::pipeline::ShaderStage::VERTEX)
        .build();

    Shadow {
        texture,
        render_target,
        view_data_buffer: view_data,
        view_data_desc_set: sh_view_data_set,
        extent,
    }
}

pub fn get_shadow_pipeline_for(
    renderer: &mut Renderer,
    world: &World,
    mesh: &super::Mesh,
) -> Result<Handle<GraphicsPipeline>, super::MaterialError> {
    // TODO: Less World, more asking the caller to provide the data
    let shader_compiler = world.read_resource::<super::shader::ShaderCompiler>();
    let frame_data = world.read_resource::<super::FrameResources>();

    let vertex_format_size = mesh.cpu_vertex_buffer.format().size();
    let shadow_vertex_format = trekant::vertex::VertexFormat::builder()
        .add_attribute(trekant::util::Format::FLOAT3) // pos
        .skip(vertex_format_size - trekant::util::Format::FLOAT3.size())
        .build();

    let descriptor = shadow_pipeline_desc(&shader_compiler, shadow_vertex_format)?;
    Ok(renderer.create_gfx_pipeline(descriptor, &frame_data.shadow.render_pass)?)
}

const NUM_SHADOW_MATRICES: u32 =
    uniform::SPOTLIGHT_SHADOW_MAP_COUNT + uniform::DIRECTIONAL_SHADOW_MAP_COUNT;

pub fn setup_shadow_resources(
    shader_compiler: &super::shader::ShaderCompiler,
    renderer: &mut Renderer,
) -> ShadowResources {
    const SPOTLIGHT_SHADOW_MAP_EXTENT: Extent2D = Extent2D {
        width: 1024,
        height: 1024,
    };

    const DIRECTIONAL_LIGHT_SHADOW_MAP_EXTENT: Extent2D = Extent2D {
        width: 4096,
        height: 4096,
    };

    let shadow_render_pass = create_shadow_render_pass(renderer);
    let view_data = [uniform::PosOnlyViewData {
        view_proj: uniform::Mat4::default(),
    }; NUM_SHADOW_MATRICES as usize];
    let view_data = UniformBufferDescriptor::from_slice(&view_data, BufferMutability::Mutable);

    let shadow_matrices_buf = renderer
        .create_resource_blocking(view_data)
        .expect("Failed to create buffer for shadow view and projection matrices");

    let directional = build_single_shadow(
        renderer,
        BufferHandle::sub_buffer(shadow_matrices_buf, 0, 1),
        shadow_render_pass,
        DIRECTIONAL_LIGHT_SHADOW_MAP_EXTENT,
    );

    let n_spotlights = uniform::SPOTLIGHT_SHADOW_MAP_COUNT;
    let mut spotlights = Vec::with_capacity(n_spotlights as usize);
    for buf_i in 1..1 + n_spotlights {
        spotlights.push(build_single_shadow(
            renderer,
            BufferHandle::sub_buffer(shadow_matrices_buf, buf_i as u32, 1),
            shadow_render_pass,
            SPOTLIGHT_SHADOW_MAP_EXTENT,
        ));
    }

    // TODO: Init pointlights here

    let shadow_dummy_pipeline = {
        let pos_only_vertex_format = VertexFormat::from(trekant::Format::FLOAT3);
        let pipeline_desc = shadow_pipeline_desc(shader_compiler, pos_only_vertex_format)
            .expect("Failed to create graphics pipeline descriptor for shadows");
        renderer
            .create_gfx_pipeline(pipeline_desc, &shadow_render_pass)
            .expect("Failed to create pipeline for shadow")
    };

    ShadowResources {
        render_pass: shadow_render_pass,
        dummy_pipeline: shadow_dummy_pipeline,
        spotlights,
        shadow_matrices_buf,
        directional,
    }
}

pub struct ShadowPassOutput {
    pub shadow_matrices: [uniform::Mat4; MAX_NUM_LIGHTS as usize],
    pub count: usize,
}

fn transition_unused_map(
    cmdbuf: &mut CommandBuffer,
    frame: &mut trekant::Frame,
    texture: Handle<trekant::Texture>,
) {
    let vk_image = frame
        .get_texture(&texture)
        .expect("Failed to get shadow texture for mem barrier")
        .vk_image();
    let barrier = vk::ImageMemoryBarrier {
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image: *vk_image,
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
    };
    cmdbuf.pipeline_barrier(
        &[barrier],
        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    );
}

pub fn shadow_pass(
    world: &World,
    frame: &mut trekant::Frame,
    shadow_resources: &ShadowResources,
    mut cmd_buffer: CommandBuffer,
) -> (CommandBuffer, ShadowPassOutput) {
    const SHADOW_CLEAR_VALUES: [vk::ClearValue; 1] = [vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    }];

    let shadow_bounds_ws =
        compute_world_shadow_bounds(world).expect("Failed to compute bounds for shadows");

    {
        debug_shadow_bounds(world, shadow_bounds_ws);
    }

    let lights = world.read_storage::<Light>();
    let transforms = world.read_storage::<Transform>();
    let entities = world.read_resource::<EntitiesRes>();

    // Note that we are using two buffers of shadow matrices:
    // 1. The buffer that is used for the shadow passes as the view proj matrix. This is allocated upfront to the max number of shadowing lights.
    // 2. The buffer in the output of this pass: This is the shadow passes that were actually run and that may be used later by the light pass.
    // Each ShadowMap component has an index into 2.
    // TODO: Can we simplify the above? E.g. by late creation and binding of desc set
    let mut shadow_matrices = [super::uniform::Mat4::default(); MAX_NUM_LIGHTS as usize];
    let mut shadow_render_info: [Option<ShadowRenderPassInfo>; MAX_NUM_LIGHTS as usize] =
        [None; MAX_NUM_LIGHTS];

    const MAX_NUM_SPOTLIGHTS: usize = if MAX_NUM_LIGHTS > 0 {
        MAX_NUM_LIGHTS - 1
    } else {
        0
    };
    let mut n_shadows = 0;
    let mut n_spotlights = 0;
    let mut found_directional_light = false;

    // Collect rendering info.
    for (e, light, tfm) in (&entities, &lights, &transforms).join() {
        if n_shadows >= MAX_NUM_LIGHTS {
            log::warn!("Too many punctual lights, skipping remaining");
            break;
        }

        let to_lightspace = Mat4::from(*tfm).inverted();
        let (proj, render_info) = match light {
            Light::Spot { angle, range, .. } => {
                if n_spotlights > MAX_NUM_SPOTLIGHTS {
                    log::warn!("Too many spotlights, won't generate shadows for all of them (max is {MAX_NUM_SPOTLIGHTS})");
                    break;
                }
                let proj = perspective_vk(angle * 2.0, 1.0, range.start, range.end);

                let idx = n_spotlights;
                n_spotlights += 1;
                let spot = &shadow_resources.spotlights[idx];
                (
                    proj,
                    ShadowRenderPassInfo {
                        extent: spot.extent,
                        render_target: spot.render_target,
                        view_data: spot.view_data_desc_set,
                        view_data_buf: spot.view_data_buffer,
                        shadow_map: ShadowMap {
                            matrix_idx: n_shadows as u32,
                            texture_idx: idx as u32,
                            shadow_type: ShadowType::Spot,
                        },
                        light_entity: e,
                    },
                )
            }
            Light::Directional { .. } => {
                if found_directional_light {
                    log::warn!("Found more than one directional light, skipping all but first");
                    continue;
                }
                found_directional_light = true;

                let dir = &shadow_resources.directional;

                let light_bounds_ls = to_lightspace * shadow_bounds_ws;
                let aabb = compute_directional_shadow_bounds(light_bounds_ls, dir.extent);
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

                (
                    proj,
                    ShadowRenderPassInfo {
                        extent: dir.extent,
                        render_target: dir.render_target,
                        view_data: dir.view_data_desc_set,
                        view_data_buf: dir.view_data_buffer,
                        shadow_map: ShadowMap {
                            matrix_idx: n_shadows as u32,
                            shadow_type: ShadowType::Directional,
                            texture_idx: 0,
                        },
                        light_entity: e,
                    },
                )
            }
            // START HERE:
            // 1. Create 6 render passes per point light
            // 2. Make the render pass computation dynamic
            // 3. Each light has to queue its own render pass, I think
            Light::Point { .. } => continue,
            Light::Ambient { .. } => continue,
        };

        // The shadow_map stores the index into the output shadow matrices buffer
        // but the shadow pass will use this shadow matrix instead so write the proj matrix there.
        shadow_matrices[render_info.view_data_buf.idx() as usize] =
            (proj * to_lightspace).into_col_array();
        shadow_render_info[n_shadows] = Some(render_info);
        n_shadows += 1;
    }

    frame
        .update_uniform_blocking(&shadow_resources.shadow_matrices_buf, &shadow_matrices)
        .expect("Failed to update matrices for shadow coords");

    // TODO: Push all unused into a vec to iterate over?

    let mut shadow_maps = world.write_storage::<ShadowMap>();

    assert_eq!(
        n_shadows,
        n_spotlights + if found_directional_light { 1 } else { 0 }
    );

    let mut output = ShadowPassOutput {
        shadow_matrices: Default::default(),
        count: n_shadows,
    };

    // Render passes
    for i in 0..n_shadows {
        let ShadowRenderPassInfo {
            extent,
            render_target,
            view_data,
            shadow_map,
            light_entity,
            view_data_buf,
        } = shadow_render_info[i].unwrap();

        let mut shadow_rp = frame
            .begin_render_pass(
                cmd_buffer,
                &shadow_resources.render_pass,
                &render_target,
                extent,
                &SHADOW_CLEAR_VALUES,
            )
            .expect("Failed to shadow begin render pass");

        shadow_rp
            .bind_graphics_pipeline(&shadow_resources.dummy_pipeline)
            .bind_shader_resource_group(0u32, &view_data, &shadow_resources.dummy_pipeline);
        super::draw_entities(world, &mut shadow_rp, super::DrawMode::ShadowsOnly);
        cmd_buffer = shadow_rp.end().expect("Failed to end shadow render pass");

        output.shadow_matrices[i] = shadow_matrices[view_data_buf.idx() as usize];
        shadow_maps
            .insert(light_entity, shadow_map)
            .expect("Failed to add shadow map for light entity");
    }

    for spotlight in shadow_resources.spotlights.iter().skip(n_spotlights) {
        transition_unused_map(&mut cmd_buffer, frame, spotlight.texture);
    }

    if !found_directional_light {
        transition_unused_map(&mut cmd_buffer, frame, shadow_resources.directional.texture);
    }

    (cmd_buffer, output)
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
        .update_uniform_blocking(
            &frame_resources.engine_shader_resources.lighting_data,
            &lighting_data,
        )
        .expect("Failed to update uniform for lighting data");

    {
        let shadow_data = uniform::ShadowData {
            matrices: shadow_pass_output.shadow_matrices,
            count: [
                shadow_pass_output.count as u32,
                u32::MAX,
                u32::MAX,
                u32::MAX,
            ],
        };

        // TODO: Merge into LightingData?
        frame
            .update_uniform_blocking(
                &frame_resources.engine_shader_resources.shadow_data,
                &shadow_data,
            )
            .expect("Failed to write the ShadowData uniform during the lighting pass");
    }
}
