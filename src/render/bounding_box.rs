use crate::common::Name;
use crate::ecs::prelude::*;
use crate::graph::sys as graph;
use crate::math::{BoundingBox, Rgba, Transform, Vec3};

use super::mesh::CpuMesh;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct RenderBoundingBox;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct BoundingBoxRenderer;

pub struct CreateRenderedBoundingBoxes;
impl<'a> System<'a> for CreateRenderedBoundingBoxes {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, BoundingBox>,
        WriteStorage<'a, RenderBoundingBox>,
        WriteStorage<'a, graph::Children>,
        WriteStorage<'a, graph::Parent>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, Name>,
        WriteStorage<'a, BoundingBoxRenderer>,
        WriteStorage<'a, CpuMesh>,
        WriteStorage<'a, super::material::Unlit>,
    );

    fn run(
        &mut self,
        (
            entities,
            bounding_box_storage,
            command_markers,
            mut children_storage,
            mut parent_storage,
            mut transforms,
            mut names,
            mut renderer_markers,
            mut meshes,
            mut materials,
        ): Self::SystemData,
    ) {
        for (ent, bbox, _) in (&entities, &bounding_box_storage, &command_markers).join() {
            let mut found = false;
            graph::breadth_first(&children_storage, ent, |node| {
                found = renderer_markers.get(node).is_some()
            });
            if found {
                continue;
            }
            let dims = bbox.max - bbox.min;
            let (vertex_buffer, index_buffer) = super::geometry::box_mesh(dims.x, dims.y, dims.z);
            let mesh = CpuMesh {
                vertex_buffer,
                index_buffer,
                polygon_mode: trekanten::pipeline::PolygonMode::Line,
            };

            let material = super::material::Unlit {
                color: Rgba::new(1.0, 0.0, 0.0, 1.0),
            };

            let mut tfm = Transform::identity();
            tfm.position = Vec3 {
                x: dims.x / 2.0,
                y: dims.y / 2.0,
                z: dims.z / 2.0,
            };

            let child = entities
                .build_entity()
                .with(Name::from("BoundingBoxRenderer"), &mut names)
                .with(tfm, &mut transforms)
                .with(BoundingBoxRenderer, &mut renderer_markers)
                .with(mesh, &mut meshes)
                .with(material, &mut materials)
                .build();

            graph::add_edge(&mut children_storage, &mut parent_storage, ent, child);
        }

        for (ent, _marker) in (&entities, &renderer_markers).join() {
            if let Some(graph::Parent { parent }) = parent_storage.get(ent) {
                if !command_markers.get(*parent).is_some() {
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
        CreateRenderedBoundingBoxes,
        std::any::type_name::<CreateRenderedBoundingBoxes>(),
        &[crate::render::debug_window::ApplySettings::ID],
    )
}
