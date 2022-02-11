use crate::ecs::prelude::*;

use crate::graph::{sys::add_edge, sys::breadth_first, Children, Parent};
use crate::render::geometry;
use crate::render::light::Light;
use crate::render::material::Unlit;
use crate::render::mesh::Mesh;

use crate::math::{Quat, Rgba, Transform, Vec3};

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct RenderLightVolume;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct LightVolumeRenderer;

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
        WriteStorage<'a, Unlit>,
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
                Light::Point { range, color } => {
                    (geometry::sphere_mesh(*range), color, Transform::identity())
                }
                Light::Spot {
                    color,
                    angle,
                    range,
                } => {
                    let radius = angle.tan() * range;
                    let mesh = geometry::cone_mesh(radius, *range);
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

            let material = Unlit {
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

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(
        RenderLightVolumes,
        std::any::type_name::<RenderLightVolumes>(),
        &[crate::render::debug::window::ApplySettings::ID],
    )
}
