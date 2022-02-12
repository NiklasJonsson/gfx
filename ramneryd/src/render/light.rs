use crate::ecs::prelude::*;
use crate::math::{
    orthographic_vk, perspective_vk, Aabb, FrustrumPlanes, Mat4, Obb, Rgb, Transform, Vec2, Vec3,
};

use trekanten::CommandBuffer;

use crate::render::uniform::{LightingData, PackedLight, ShadowMatrices, ViewData, MAX_NUM_LIGHTS};
use std::ops::Range;

#[derive(
    Component, ramneryd_derive::Visitable, serde::Serialize, serde::Deserialize, Clone, Debug,
)]
pub enum Light {
    // Range is the radius of the sphere
    Point {
        color: Rgb,
        range: f32,
    },
    Directional {
        color: Rgb,
    },
    /// `angle` is from the center line of the cone.
    /// `range` the start and end of the cone.
    Spot {
        color: Rgb,
        angle: f32,
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

/// Compute the bounds of the view are that we want to cast shadows on.
/// The coordinates are in world-space.
fn compute_shadow_bounds(world: &World) -> Option<Obb> {
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

pub fn light_and_shadow_pass(
    world: &World,
    frame: &mut trekanten::Frame,
    frame_resources: &super::FrameData,
    mut cmd_buffer: CommandBuffer,
) -> CommandBuffer {
    let light_bounds_ws =
        compute_shadow_bounds(world).expect("Failed to compute bounds for shadows");

    use trekanten::raw_vk;
    let mut lighting_data = LightingData::default();
    let mut shadow_matrices = ShadowMatrices::default();

    let clear_values = [raw_vk::ClearValue {
        depth_stencil: raw_vk::ClearDepthStencilValue {
            depth: 1.0,
            stencil: 0,
        },
    }];

    let lights = world.read_storage::<Light>();
    let transforms = world.read_storage::<Transform>();
    let mut n_ambients = 0;

    let super::FrameData {
        shadow:
            super::ShadowData {
                render_pass,
                dummy_pipeline,
                spotlights,
            },
        ..
    } = frame_resources;

    for (idx, (light, tfm)) in (&lights, &transforms).join().enumerate() {
        if idx >= MAX_NUM_LIGHTS {
            log::warn!("Too many punctual lights, skipping remaining");
            break;
        }

        if let Light::Ambient { color, strength } = &light {
            if n_ambients > 0 {
                log::warn!("Too many ambient lights, skipping all but first");
            } else {
                lighting_data.ambient = [color.r, color.g, color.b, *strength];
                n_ambients = 1;
            }
            continue;
        }

        let direction = tfm.rotation * Light::DEFAULT_FACING;
        let light_pos = tfm.position.with_w(1.0);
        let to_lightspace = Mat4::from(*tfm).inverted();
        let (packed_light, shadow_view_data) = match light {
            Light::Spot {
                angle,
                color,
                range,
            } => {
                let proj = perspective_vk(angle * 2.0, 1.0, range.start, range.end);
                let pos = tfm.position.with_w(1.0).into_array();

                (
                    PackedLight {
                        pos,
                        dir_cutoff: [direction.x, direction.y, direction.z, angle.cos()],
                        color_range: [color.r, color.g, color.b, range.end],
                        ..Default::default()
                    },
                    Some(proj),
                )
            }
            Light::Directional { color } => {
                let (min, max) = {
                    let obb_lightspace = to_lightspace * light_bounds_ws;
                    let aabb_lightspace = Aabb::from(obb_lightspace);

                    // Ensure that the aabb only moves in texel-sized increments. This stops shadows from moving around as the
                    // camera moves around (as the shadow aabb is computed from the camera). The idea is to fix the aabb to the
                    // grid of the shadow map in space, ensuring that the rasterized objects during a shadow pass consistently
                    // overlaps the right texel. Hard to explain in text but if a triangle is drawn accross a grid. Then the grid
                    // shifts very slightly (less than a texex size) so that the triangle no longer covers the centre of a cell (texel),
                    // that depth buffer texel will no longer have the depth of the camera. If instead, the grid only shifts in texel-sized
                    // increments. The triangle will always overlap the subtexel area in the same way, even if the grid has shifted.
                    // See https://docs.microsoft.com/en-us{/windows/win32/dxtecharts/cascaded-shadow-maps for details.
                    let shadow_map_extents = Vec2 {
                        x: super::SHADOW_MAP_EXTENT.width as f32,
                        y: super::SHADOW_MAP_EXTENT.height as f32,
                    };

                    let min = aabb_lightspace.min.xy();
                    let max = aabb_lightspace.max.xy();

                    let texel_size_lightspace = (max - min) / shadow_map_extents;

                    let min = (min / texel_size_lightspace).floor() * texel_size_lightspace;
                    let max = (max / texel_size_lightspace).floor() * texel_size_lightspace;

                    (
                        min.with_z(aabb_lightspace.min.z),
                        max.with_z(aabb_lightspace.max.z),
                    )
                };

                let proj: Mat4 = orthographic_vk(FrustrumPlanes {
                    left: min.x,
                    right: max.x,
                    bottom: min.y,
                    top: max.y,
                    near: min.z,
                    far: max.z,
                });

                (
                    PackedLight {
                        pos: [0.0, 0.0, 0.0, 0.0],
                        dir_cutoff: [direction.x, direction.y, direction.z, 0.0],
                        color_range: [color.r, color.g, color.b, 0.0],
                        ..Default::default()
                    },
                    Some(proj),
                )
            }
            Light::Point { color, range } => (
                PackedLight {
                    pos: tfm.position.with_w(1.0).into_array(),
                    dir_cutoff: [0.0, 0.0, 0.0, 0.0],
                    color_range: [color.r, color.g, color.b, *range],
                    ..Default::default()
                },
                None,
            ),
            Light::Ambient { .. } => unreachable!("Should have been handled already"),
        };
        lighting_data.punctual_lights[lighting_data.num_lights[0] as usize] = packed_light;
        lighting_data.num_lights[0] += 1;

        if let Some(shadow_proj) = shadow_view_data {
            let shadow_idx = shadow_matrices.num_matrices[0];
            shadow_matrices.num_matrices[0] += 1;

            assert!(lighting_data.num_lights[0] > 0);
            lighting_data.punctual_lights[(lighting_data.num_lights[0] - 1) as usize].shadow_idx =
                [shadow_idx; 4];

            let shadow_idx = shadow_idx as usize;

            let shadow_view_data = ViewData {
                view_pos: light_pos.into_array(),
                view_proj: (shadow_proj * to_lightspace).into_col_array(),
            };

            shadow_matrices.matrices[shadow_idx] = shadow_view_data.view_proj;
            frame
                .update_uniform_blocking(
                    &spotlights[shadow_idx].view_data_buffer,
                    &shadow_view_data,
                )
                .expect("Failed to update view data for shadow pass");

            let mut shadow_rp = frame
                .begin_render_pass(
                    cmd_buffer,
                    render_pass,
                    &spotlights[shadow_idx].render_target,
                    super::SHADOW_MAP_EXTENT,
                    &clear_values,
                )
                .expect("Failed to shadow begin render pass");

            shadow_rp
                .bind_graphics_pipeline(dummy_pipeline)
                .bind_shader_resource_group(
                    0u32,
                    &spotlights[shadow_idx].view_data_desc_set,
                    dummy_pipeline,
                );
            super::draw_entities(world, &mut shadow_rp, super::DrawMode::ShadowsOnly);
            cmd_buffer = shadow_rp.end().expect("Failed to end shadow render pass");
        }
    }

    let num_shadows = shadow_matrices.num_matrices[0];

    frame
        .update_uniform_blocking(
            &frame_resources.pbr_resources.shadow_matrices_buffer,
            &shadow_matrices,
        )
        .expect("Failed to update matrices for shadow coords");
    frame
        .update_uniform_blocking(&frame_resources.pbr_resources.light_buffer, &lighting_data)
        .expect("Failed to update uniform for lighting data");

    // transistion unused images to depth stencil read optimal as this won't be done by the render pass
    // TODO(perf): Don't allocate, store a vector for reuse
    let mut barriers = Vec::with_capacity(super::NUM_SPOTLIGHT_SHADOW_MAPS - num_shadows as usize);
    for spotlight in spotlights
        .iter()
        .take(super::NUM_SPOTLIGHT_SHADOW_MAPS)
        .skip(num_shadows as usize)
    {
        let handle = spotlight.texture;
        let vk_image = frame
            .get_texture(&handle)
            .expect("Failed to get shadow texture for mem barrier")
            .vk_image();
        let barrier = raw_vk::ImageMemoryBarrier {
            old_layout: raw_vk::ImageLayout::UNDEFINED,
            new_layout: raw_vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            src_queue_family_index: raw_vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: raw_vk::QUEUE_FAMILY_IGNORED,
            image: *vk_image,
            subresource_range: raw_vk::ImageSubresourceRange {
                aspect_mask: raw_vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_access_mask: raw_vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access_mask: raw_vk::AccessFlags::SHADER_READ,
            ..Default::default()
        };
        barriers.push(barrier);
    }

    cmd_buffer.pipeline_barrier(
        &barriers,
        raw_vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        raw_vk::PipelineStageFlags::FRAGMENT_SHADER,
    );

    cmd_buffer
}
