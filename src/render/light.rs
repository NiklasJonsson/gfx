use crate::ecs::prelude::*;
use crate::math::{Quat, Transform, Vec3, Vec4};

use crate::graph::{sys::add_edge, sys::breadth_first, Children, Parent};
use crate::render::mesh::CpuMesh;
use crate::render::uniform::{LightingData, PackedLight, MAX_NUM_LIGHTS};

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct RenderLightVolume;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct LightVolumeRenderer;

#[derive(Component)]
#[component(inspect)]
pub enum Light {
    // Range is the radius of the sphere
    Point { color: Vec3, range: f32 },
    Directional { color: Vec3 },
    // Angle is from the center line of the cone & range the height of the cone
    Spot { color: Vec3, angle: f32, range: f32 },
    Ambient { color: Vec3, strength: f32 },
}

impl Light {
    // As per gltf extension, KHR_light_punctual. Also makes sense as perspective matrix is based on
    // camera looking in -z
    pub const DEFAULT_FACING: Vec3 = Vec3 { x: 0.0, y: 0.0, z: -1.0 };
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
        WriteStorage<'a, CpuMesh>,
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

            let (vertex_buffer, index_buffer, color, tfm) = match light {
                Light::Point { range, color } => {
                    let mesh = super::geometry::sphere_mesh(*range);
                    (mesh.0, mesh.1, color, Transform::identity())
                }
                Light::Spot {
                    color,
                    angle,
                    range,
                } => {
                    let radius = angle.tan() * range;
                    let (v, i) = super::geometry::cone_mesh(radius, *range);
                    // Cone mesh has base at origin, apex at (0, range, 0). We want to have apex at origin (translation) and then
                    // rotated to Light::DEFAULT_FACING
                    let translation = Transform::pos(0.0, -*range, 0.0);
                    let rotation = Transform { rotation: Quat::rotation_from_to_3d(Vec3::new(0.0, -1.0, 0.0), Light::DEFAULT_FACING), ..Default::default()};
                    let tfm = rotation * translation;
                    (v, i, color, tfm)
                }
                _ => continue,
            };

            let mesh = CpuMesh {
                vertex_buffer,
                index_buffer,
                polygon_mode: trekanten::pipeline::PolygonMode::Line,
            };

            let material = super::material::Unlit {
                color: Vec4::new(color.x, color.y, color.z, 1.0),
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
                if !command_markers.get(*parent).is_some() {
                    entities.delete(ent).unwrap();
                }
            } else {
                entities.delete(ent).unwrap();
            }
        }
    }
}

pub fn build_light_data_uniform(world: &World) -> LightingData {
    let mut data = LightingData::default();
    let lights = world.read_storage::<Light>();
    let transforms = world.read_storage::<Transform>();
    let mut n_ambients = 0;
    for (idx, (light, tfm)) in (&lights, &transforms).join().enumerate() {
        if idx >= MAX_NUM_LIGHTS {
            log::warn!("Too many punctual lights, skipping remaining");
            break;
        }
        // TODO: Ambient needs a position, is this good?
        if let Light::Ambient { color, strength } = &light {
            if n_ambients > 0 {
                log::warn!("Too many ambient lights, skipping all but first");
            } else {
                data.ambient = [color.x, color.y, color.z, *strength];
                n_ambients = 1;
            }
            continue;
        }

        data.punctual_lights[data.num_lights as usize] = match light {
            Light::Directional { color } => {
                let direction = tfm.rotation * Light::DEFAULT_FACING;
                PackedLight {
                    pos: [0.0, 0.0, 0.0, 0.0],
                    dir_cutoff: [direction.x, direction.y, direction.z, 0.0],
                    color_range: [color.x, color.y, color.z, 0.0],
                }
            }
            Light::Point { color, range } => PackedLight {
                pos: [tfm.position.x, tfm.position.y, tfm.position.z, 1.0],
                dir_cutoff: [0.0, 0.0, 0.0, 0.0],
                color_range: [color.x, color.y, color.z, *range],
            },
            Light::Spot {
                color,
                angle,
                range,
            } => {
                let direction = tfm.rotation * Light::DEFAULT_FACING;
                PackedLight {
                    pos: [tfm.position.x, tfm.position.y, tfm.position.z, 1.0],
                    dir_cutoff: [direction.x, direction.y, direction.z, angle.cos()],
                    color_range: [color.x, color.y, color.z, *range],
                }
            }
            Light::Ambient { .. } => unreachable!("Should have been handled already"),
        };
        data.num_lights += 1;
    }

    data
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(
        RenderLightVolumes,
        std::any::type_name::<RenderLightVolumes>(),
        &[crate::render::debug_window::ApplySettings::ID],
    )
}
