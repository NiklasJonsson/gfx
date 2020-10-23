use specs::prelude::*;

use crate::transform_graph;
use imgui::*;

fn build_tree<'a>(world: &World, ui: &imgui::Ui<'a>, ent: specs::Entity) {
    TreeNode::new(&im_str!("{:?}", ent)).build(&ui, || {
        if let Some(node) = world
            .read_component::<transform_graph::RenderGraphNode>()
            .get(ent)
        {
            for child in node.children.iter() {
                build_tree(world, ui, *child);
            }
        }
    });
}

pub fn build_ui<'a>(world: &World, ui: &imgui::Ui<'a>, pos: [f32; 2]) -> [f32; 2] {
    let root_storage = world.read_storage::<transform_graph::RenderGraphRoot>();
    let entities = world.read_resource::<specs::world::EntitiesRes>();

    let size = [300.0, 300.0];
    imgui::Window::new(im_str!("Scene"))
        .size(size, imgui::Condition::FirstUseEver)
        .position(pos, imgui::Condition::FirstUseEver)
        .build(&ui, || {
            for (ent, _root) in (&entities, &root_storage).join() {
                build_tree(world, ui, ent);
            }
        });
    size
}
