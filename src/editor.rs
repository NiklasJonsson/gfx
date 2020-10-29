use specs::prelude::*;

use crate::transform_graph;
use imgui::*;

fn build_tree<'a>(world: &World, ui: &imgui::Ui<'a>, ent: specs::Entity) -> Option<specs::Entity> {
    let mut inspected = None;

    TreeNode::new(&im_str!("{:?}", ent)).build(&ui, || {
        let pressed = ui.small_button(im_str!("inspect"));
        if pressed {
            inspected = Some(ent);
        }
        if let Some(node) = world
            .read_component::<transform_graph::RenderGraphNode>()
            .get(ent)
        {
            for child in node.children.iter() {
                let new = build_tree(world, ui, *child);
                inspected = inspected.or(new);
            }
        }
    });

    inspected
}

fn build_inspector<'a>(world: &World, ui: &imgui::Ui<'a>, ent: specs::Entity) {
    use crate::math::Transform;

    ui.text(im_str!("{:?}", ent));
    ui.separator();
    /*
    let transforms = world.read_component::<Transform>();
    if let Some(tfm) = transforms.get(ent) {
        ui.text(im_str!("{:#?}", tfm.));
    }
    */
}

struct SelectedEntity {
    entity: specs::Entity,
}

pub fn build_ui<'a>(world: &mut World, ui: &imgui::Ui<'a>, _pos: [f32; 2]) -> [f32; 2] {
    let [width, _height] = ui.io().display_size;
    let scene_window_size = [300.0, 500.0];
    let scene_window_pos = [width - scene_window_size[0], 0.0];

    let mut inspected: Option<specs::Entity> = None;

    {
        let root_storage = world.read_storage::<transform_graph::RenderGraphRoot>();
        let entities = world.read_resource::<specs::world::EntitiesRes>();

        imgui::Window::new(im_str!("Scene"))
            .position(scene_window_pos, Condition::Always)
            .size(scene_window_size, Condition::Always)
            .build(&ui, || {
                for (ent, _root) in (&entities, &root_storage).join() {
                    inspected = build_tree(world, ui, ent);
                }
            });

        if world.has_value::<SelectedEntity>() && inspected.is_none() {
            inspected = Some(world.read_resource::<SelectedEntity>().entity);
        }
    }

    let inspected_window_size = [scene_window_size[0], 300.0];
    let inspected_window_pos = [scene_window_pos[0], scene_window_size[1]];
    if let Some(ent) = inspected {
        imgui::Window::new(im_str!("Inspector"))
            .position(inspected_window_pos, Condition::Always)
            .size(inspected_window_size, Condition::Always)
            .build(&ui, || {
                build_inspector(world, ui, ent);
            });
        world.insert(SelectedEntity { entity: ent });
    }

    [0.0, 0.0]
}
