use crate::ecs::prelude::*;
use crate::math::{Transform, Vec3};

use crate::graph::{sys::add_edge, sys::breadth_first, Children, Parent};
use crate::render::material::{Material, PendingMaterial};
use crate::render::mesh::PendingMesh;
use crate::render::uniform::{LightingData, PackedLight, MAX_NUM_LIGHTS};

use trekanten::loader::ResourceLoader;
use trekanten::uniform::OwningUniformBufferDescriptor;
use trekanten::BufferMutability;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct RenderLightVolume;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct LightVolumeRenderer;

#[derive(Component)]
#[component(inspect)]
pub enum Light {
    Point { color: Vec3, range: f32 },
    Directional { color: Vec3 },
    Spot { color: Vec3, angle: f32, range: f32 },
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
        WriteStorage<'a, PendingMesh>,
        WriteStorage<'a, PendingMaterial>,
        WriteExpect<'a, trekanten::Loader>,
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
            loader,
        ) = data;

        for (ent, light, _) in (&entities, &lights, &command_markers).join() {
            let (radius, color) = if let Light::Point { range, color } = light {
                (*range, *color)
            } else {
                continue;
            };

            let mut found_child = false;
            breadth_first(&children, ent, |node| {
                found_child = renderer_markers.get(node).is_some()
            });
            if found_child {
                continue;
            }

            let (vertices, indices) = super::geometry::sphere_mesh(radius);
            let vertex_buffer = loader.load(vertices);
            let index_buffer = loader.load(indices);
            let pending_mesh = PendingMesh(Some(super::GpuMesh {
                mesh: trekanten::mesh::Mesh {
                    vertex_buffer,
                    index_buffer,
                },
                polygon_mode: trekanten::pipeline::PolygonMode::Line,
            }));

            let uniform_data = super::uniform::UnlitUniformData {
                color: [color.x, color.y, color.z, 1.0],
            };

            let color_uniform = loader.load(OwningUniformBufferDescriptor::from_vec(
                vec![uniform_data],
                BufferMutability::Immutable,
            ));

            let pending_material = PendingMaterial::from(Material::Unlit { color_uniform });

            let child = entities
                .build_entity()
                .with(Transform::identity(), &mut transforms)
                .with(LightVolumeRenderer, &mut renderer_markers)
                .with(pending_mesh, &mut meshes)
                .with(pending_material, &mut materials)
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
    for (idx, (light, tfm)) in (&lights, &transforms).join().enumerate() {
        if idx >= MAX_NUM_LIGHTS {
            log::warn!("Too many punctual lights, skipping remaining");
            break;
        }
        data.punctual_lights[idx] = match light {
            Light::Directional { color } => {
                let direction = tfm.rotation * Vec3::new(0.0, -1.0, 0.0);
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
                let direction = tfm.rotation * Vec3::new(0.0, -1.0, 0.0);
                PackedLight {
                    pos: [tfm.position.x, tfm.position.y, tfm.position.z, 1.0],
                    dir_cutoff: [direction.x, direction.y, direction.z, angle.cos()],
                    color_range: [color.x, color.y, color.z, *range],
                }
            }
        };
        data.num_lights = (idx + 1) as u32;
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
