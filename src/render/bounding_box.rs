use crate::common::Name;
use crate::ecs::prelude::*;
use crate::graph;
use crate::math::{BoundingBox, Transform, Vec3};
use crate::render::{material::Material, material::PendingMaterial, mesh::PendingMesh};

use trekanten::loader::ResourceLoader;
use trekanten::uniform::OwningUniformBufferDescriptor;
use trekanten::BufferMutability;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct DoRenderBoundingBox;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct DontRenderBoundingBox;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct BoundingBoxRenderer;

pub struct CreateRenderedBoundingBoxes;
impl<'a> System<'a> for CreateRenderedBoundingBoxes {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, BoundingBox>,
        WriteStorage<'a, DoRenderBoundingBox>,
        WriteStorage<'a, DontRenderBoundingBox>,
        WriteStorage<'a, graph::Children>,
        WriteStorage<'a, graph::Parent>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, Name>,
        WriteStorage<'a, BoundingBoxRenderer>,
        WriteStorage<'a, super::mesh::PendingMesh>,
        WriteStorage<'a, super::material::PendingMaterial>,
        WriteExpect<'a, trekanten::Loader>,
    );

    fn run(
        &mut self,
        (
            entities,
            bounding_box_storage,
            mut do_render_bounding_box,
            mut dont_render_bounding_box,
            mut children_storage,
            mut parent_storage,
            mut transforms,
            mut names,
            mut bounding_box_renderer_storage,
            mut meshes,
            mut materials,
            loader,
        ): Self::SystemData,
    ) {
        // TODO: Create one big buffer for each of vertex/index/uniform
        for (ent, bbox, _) in (&entities, &bounding_box_storage, &do_render_bounding_box).join() {
            let dims = bbox.max - bbox.min;
            let (vertices, indices) = super::geometry::box_mesh(dims.x, dims.y, dims.z);
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
                color: [1.0, 0.0, 0.0, 1.0],
            };

            let color_uniform = loader.load(OwningUniformBufferDescriptor::from_vec(
                vec![uniform_data],
                BufferMutability::Immutable,
            ));

            let pending_material = PendingMaterial::from(Material::Unlit { color_uniform });

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
                .with(BoundingBoxRenderer, &mut bounding_box_renderer_storage)
                .with(pending_mesh, &mut meshes)
                .with(pending_material, &mut materials)
                .build();

            graph::add_edge_sys(&mut children_storage, &mut parent_storage, ent, child);
        }

        do_render_bounding_box.clear();

        for (ent, _) in (&entities, &dont_render_bounding_box).join() {
            graph::breadth_first_sys(&children_storage, ent, |n| {
                if bounding_box_renderer_storage.get(n).is_some() {
                    entities.delete(n).expect("bad graph");
                }
            });
        }
        dont_render_bounding_box.clear();
    }
}

// TODO: Remaining:
// * keypress/ui assigns DeleteRenderBox
// * Assign bounding box during mesh loading
// * button in inspector to render bbox
// * implement geometry::box

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(
        CreateRenderedBoundingBoxes,
        std::any::type_name::<CreateRenderedBoundingBoxes>(),
        &[],
    )
}
