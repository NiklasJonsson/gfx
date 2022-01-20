use crate::ecs::prelude::*;
use crate::math::{
    orthographic_vk, perspective_vk, Aabb, FrustrumPlanes, Mat4, Obb, Quat, Rgb, Rgba, Transform,
    Vec3,
};

use trekanten::CommandBuffer;

use crate::graph::{sys::add_edge, sys::breadth_first, Children, Parent};
use crate::render::mesh::Mesh;
use crate::render::uniform::{LightingData, PackedLight, ShadowMatrices, ViewData, MAX_NUM_LIGHTS};

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct RenderLightVolume;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct LightVolumeRenderer;

#[derive(
    Component, ramneryd_derive::Visitable, serde::Serialize, serde::Deserialize, Clone, Debug,
)]
pub enum Light {
    // Range is the radius of the sphere
    Point { color: Rgb, range: f32 },
    Directional { color: Rgb },
    // Angle is from the center line of the cone & range the height of the cone
    Spot { color: Rgb, angle: f32, range: f32 },
    Ambient { color: Rgb, strength: f32 },
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
            range: 5.0,
        }
    }
}

pub struct RenderLightVolumes;

impl<'a> System<'a> for RenderLightVolumes {
    type SystemData = (
        ReadStorage<'a, Light>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, Parent>,
        WriteStorage<'a, Children>,
        Entities<'a>,
        ReadStorage<'a, RenderLightVolume>,
        WriteStorage<'a, LightVolumeRenderer>,
        WriteStorage<'a, Mesh>,
        WriteStorage<'a, super::material::Unlit>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (
            lights,
            mut transforms,
            mut parents,
            mut children,
            entities,
            command_markers,
            mut renderer_markers,
            mut meshes,
            mut materials,
        ) = data;

        for (ent, light, _) in (&entities, &lights, &command_markers).join() {
            let mut found_child = false;
            breadth_first(&children, ent, |node| {
                found_child = renderer_markers.get(node).is_some()
            });
            if found_child {
                continue;
            }

            let (mesh, color, tfm) = match light {
                Light::Point { range, color } => (
                    super::geometry::sphere_mesh(*range),
                    color,
                    Transform::identity(),
                ),
                Light::Spot {
                    color,
                    angle,
                    range,
                } => {
                    let radius = angle.tan() * range;
                    let mesh = super::geometry::cone_mesh(radius, *range);
                    // Cone mesh has base at origin, apex at (0, range, 0). We want to have apex at origin (translation) and then
                    // rotated to Light::DEFAULT_FACING
                    let translation = Transform::pos(0.0, -*range, 0.0);
                    let rotation = Transform {
                        rotation: Quat::rotation_from_to_3d(
                            Vec3::new(0.0, -1.0, 0.0),
                            Light::DEFAULT_FACING,
                        ),
                        ..Default::default()
                    };
                    let tfm = rotation * translation;
                    (mesh, color, tfm)
                }
                _ => continue,
            };

            let material = super::material::Unlit {
                color: Rgba::from_opaque(*color),
                polygon_mode: trekanten::pipeline::PolygonMode::Line,
            };

            let child = entities
                .build_entity()
                .with(tfm, &mut transforms)
                .with(LightVolumeRenderer, &mut renderer_markers)
                .with(mesh, &mut meshes)
                .with(material, &mut materials)
                .build();

            add_edge(&mut children, &mut parents, ent, child);
        }
        for (ent, _marker) in (&entities, &renderer_markers).join() {
            if let Some(Parent { parent }) = parents.get(ent) {
                if command_markers.get(*parent).is_none() {
                    entities.delete(ent).unwrap();
                }
            } else {
                entities.delete(ent).unwrap();
            }
        }
    }
}

pub fn light_and_shadow_pass(
    world: &World,
    frame: &mut trekanten::Frame,
    frame_resources: &super::FrameData,
    light_bounds_ws: Obb,
    mut cmd_buffer: CommandBuffer,
) -> CommandBuffer {
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
                range,
                color,
            } => {
                let proj = perspective_vk(angle * 2.0, 1.0, 1.0, *range);
                let pos = tfm.position.with_w(1.0).into_array();

                (
                    PackedLight {
                        pos,
                        dir_cutoff: [direction.x, direction.y, direction.z, angle.cos()],
                        color_range: [color.r, color.g, color.b, *range],
                        ..Default::default()
                    },
                    Some(proj),
                )
            }
            Light::Directional { color } => {
                let (min, max) = {
                    let obb_lightspace = to_lightspace * light_bounds_ws;
                    let aabb_lightspace = Aabb::from(obb_lightspace);
                    (aabb_lightspace.min, aabb_lightspace.max)
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

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(
        RenderLightVolumes,
        std::any::type_name::<RenderLightVolumes>(),
        &[crate::render::debug_window::ApplySettings::ID],
    )
}
