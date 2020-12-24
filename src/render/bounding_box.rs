use crate::common::Name;
use crate::ecs::prelude::*;
use crate::graph;
use crate::math::{BoundingBox, ModelMatrix, Transform};
use crate::render::geometry::Mesh;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct DoRenderBoundingBox;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct BoundingBoxRenderer;

pub struct CreateRenderedBoundingBoxes;
impl<'a> System<'a> for CreateRenderedBoundingBoxes {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, BoundingBox>,
        ReadStorage<'a, ModelMatrix>,
        WriteStorage<'a, DoRenderBoundingBox>,
        WriteStorage<'a, graph::Children>,
        WriteStorage<'a, graph::Parent>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, Name>,
        WriteStorage<'a, BoundingBoxRenderer>,
        WriteStorage<'a, Mesh>,
    );

    fn run(
        &mut self,
        (
            entities,
            bounding_box_storage,
            matrices,
            mut do_render_bounding_box,
            mut children_storage,
            mut parent_storage,
            mut transforms,
            mut names,
            mut bounding_box_renderer_storage,
            mut meshes,
        ): Self::SystemData,
    ) {
        for (ent, bbox, matrix, _) in (
            &entities,
            &bounding_box_storage,
            &matrices,
            &do_render_bounding_box,
        )
            .join()
        {
            let mut total_bbox = *matrix * *bbox;
            let update_bbox = |e: Entity| {
                if let Some(bbox) = bounding_box_storage.get(e) {
                    let m = *matrices.get(ent).expect("No model matrix for bounding box");
                    total_bbox.combine(m * *bbox);
                }
            };

            graph::breadth_first_sys(&children_storage, ent, update_bbox);
            let child = entities.create();
            graph::add_edge_sys(&mut children_storage, &mut parent_storage, ent, child);
            names
                .insert(child, Name::from("RendererBoundingBox"))
                .expect("Failed to add component");
            transforms
                .insert(child, Transform::identity())
                .expect("Failed to insert transform");
            bounding_box_renderer_storage
                .insert(child, BoundingBoxRenderer {})
                .expect("Failed to insert bb renderer");
            meshes
                .insert(child, super::geometry::box_mesh())
                .expect("Failed to insert mesh");
        }

        do_render_bounding_box.clear();
    }
}

/*
let remove_rbbs = !settings.render_bounding_box;

if remove_rbbs {
    let mut to_remove = Vec::new();
    for (root_ent, bb) in (&entities, &rbbs).join() {
        let bb_node = bb.0;
        meshes.remove(bb_node);
        renderables.remove(bb_node);
        to_remove.push(root_ent);
    }

    for ent in to_remove {
        rbbs.remove(ent);
    }

    return;
}

for (root_ent, _root) in (&entities, &roots).join() {

    let ty = MeshType::Line { indices };

    meshes
        .insert(child, mesh)
        .expect("Unable to insert bb mesh");
    materials
        .insert(child, material)
        .expect("Unable to add bb material");
*/

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(
        CreateRenderedBoundingBoxes,
        std::any::type_name::<CreateRenderedBoundingBoxes>(),
        &[],
    )
}
