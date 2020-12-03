use specs::prelude::*;

use crate::common::Name;
use crate::graph;
use crate::math::ModelMatrix;
use crate::math::Transform;
use imgui::*;

fn name(world: &World, ent: Entity) -> String {
    let names = world.read_component::<Name>();
    let name: &str = names.get(ent).map(|n| n.0.as_str()).unwrap_or("");
    format!("{} ({}, {})", name, ent.id(), ent.gen().id())
}

fn build_tree<'a>(world: &World, ui: &imgui::Ui<'a>, ent: specs::Entity) -> Option<specs::Entity> {
    let mut inspected = None;

    let name = im_str!("{}", name(world, ent));
    TreeNode::new(&name).build(&ui, || {
        let pressed = ui.small_button(im_str!("inspect"));
        if pressed {
            inspected = Some(ent);
        }
        if let Some(node) = world.read_component::<graph::Children>().get(ent) {
            for child in node.children.iter() {
                let new = build_tree(world, ui, *child);
                inspected = inspected.or(new);
            }
        }
    });

    inspected
}

fn build_inspector<'a>(world: &mut World, ui: &imgui::Ui<'a>, ent: specs::Entity) {
    use crate::render::ReloadMaterial;

    ui.text(im_str!("{}", name(world, ent)));
    ui.separator();
    let pressed = ui.small_button(im_str!("reload material"));
    if pressed {
        world
            .write_component::<ReloadMaterial>()
            .insert(ent, ReloadMaterial {})
            .expect("Failed to write!");
    }
    ui.separator();
    let transforms = world.read_component::<Transform>();
    let tfm = transforms.get(ent);
    let extra = if tfm.is_none() { " (None)" } else { "" };
    if CollapsingHeader::new(&im_str!("Transform{}", extra)).build(ui) {
        if let Some(tfm) = tfm {
            let mut pos = tfm.position.into_array();
            InputFloat3::new(ui, im_str!("Position"), &mut pos)
                .read_only(true)
                .build();

            let mut rot = tfm.rotation.into_vec4().into_array();
            InputFloat4::new(ui, im_str!("Rotation"), &mut rot)
                .read_only(true)
                .build();

            let mut scale = tfm.scale;
            InputFloat::new(ui, im_str!("Scale"), &mut scale)
                .read_only(true)
                .build();
        }
    }

    let matrices = world.read_component::<ModelMatrix>();
    let mat = matrices.get(ent);
    let extra = if mat.is_none() { " (None)" } else { "" };
    if CollapsingHeader::new(&im_str!("ModelMatrix{}", extra)).build(ui) {
        if let Some(mat) = mat {
            let rows = mat.0.into_row_arrays();
            for r in rows.iter() {
                let mut row = *r;
                InputFloat4::new(ui, im_str!(""), &mut row)
                    .read_only(true)
                    .build();
            }
        }
    }
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
        let parent_storage = world.read_storage::<graph::Parent>();
        let entities = world.read_resource::<specs::world::EntitiesRes>();

        imgui::Window::new(im_str!("Scene"))
            .position(scene_window_pos, Condition::Always)
            .size(scene_window_size, Condition::Always)
            .build(&ui, || {
                for (ent, _root) in (&entities, !&parent_storage).join() {
                    inspected = inspected.or(build_tree(world, ui, ent));
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
