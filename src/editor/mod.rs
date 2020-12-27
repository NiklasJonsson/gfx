use specs::prelude::*;

use crate::common::Name;
use crate::ecs;
use crate::graph;
use imgui::*;

pub(crate) mod inspect;
pub use inspect::Inspect;

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
        if let Some(children) = world.read_component::<graph::Children>().get(ent) {
            for child in children.iter() {
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

    for comp in ecs::meta::ALL_COMPONENTS {
        if (comp.has)(world, ent) {
            if comp.size == 0 {
                let _open = CollapsingHeader::new(&imgui::ImString::from(String::from(comp.name)))
                    .leaf(true)
                    .build(ui);
            } else if let Some(inspect) = comp.inspect {
                inspect(world, ent, ui);
            } else if CollapsingHeader::new(&imgui::ImString::from(String::from(comp.name)))
                .build(ui)
            {
                ui.text(im_str!("unimplemented"));
            }
        }
    }
}

struct SelectedEntity {
    entity: specs::Entity,
}

pub fn build_ui<'a>(world: &mut World, ui: &imgui::Ui<'a>) {
    let dt = *world.read_resource::<crate::time::DeltaTime>();
    let size = [400.0, 300.0];
    let pos = [0.0, 0.0];
    imgui::Window::new(im_str!("Overview"))
        .size(size, imgui::Condition::FirstUseEver)
        .position(pos, imgui::Condition::FirstUseEver)
        .build(&ui, || {
            ui.text(im_str!("FPS: {:.3}", dt.as_fps()));
            let mut p = crate::render::camera_pos(world).into_array();

            InputFloat3::new(ui, im_str!("Camera pos"), &mut p)
                .read_only(true)
                .build();
            ui.text(im_str!("#components: {}", ecs::meta::ALL_COMPONENTS.len()));
        });

    {
        let mut y_offset = 0.0;
        let funcs = [
            crate::render::ui::build_ui,
            crate::settings::build_ui,
            crate::game_state::build_ui,
            crate::io::input::build_ui,
            crate::profile::build_ui,
        ];
        for func in funcs.iter() {
            let size = func(world, &ui, [0.0, y_offset]);
            y_offset += size[1];
        }
    }

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
                    inspected = inspected.or_else(|| build_tree(world, ui, ent));
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
            .position(inspected_window_pos, Condition::FirstUseEver)
            .size(inspected_window_size, Condition::FirstUseEver)
            .build(&ui, || {
                build_inspector(world, ui, ent);
            });
        world.insert(SelectedEntity { entity: ent });
    }
}
