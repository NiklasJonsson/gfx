use specs::prelude::*;

use crate::common::Name;
use crate::graph;
use imgui::{Condition, InputFloat3, TreeNode};

pub mod inspector;

fn name(world: &World, ent: Entity) -> String {
    let names = world.read_component::<Name>();
    let name: &str = names.get(ent).map(|n| n.0.as_str()).unwrap_or("");
    format!("{} ({}, {})", name, ent.id(), ent.gen().id())
}

fn build_tree<'a>(
    world: &World,
    ui: &crate::render::ui::UiFrame<'a>,
    ent: specs::Entity,
) -> Option<specs::Entity> {
    let mut inspected = None;

    let name = name(world, ent);
    TreeNode::new(&name).build(ui.inner(), || {
        let pressed = ui.inner().small_button("inspect");
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

fn build_inspector<'a>(world: &mut World, ui: &crate::render::ui::UiFrame<'a>, ent: specs::Entity) {
    use crate::render::ReloadMaterial;

    ui.inner().text(name(world, ent));
    ui.inner().separator();
    let pressed = ui.inner().small_button("reload material");
    if pressed {
        world
            .write_component::<ReloadMaterial>()
            .insert(ent, ReloadMaterial {})
            .expect("Failed to write!");
    }
    ui.inner().separator();

    let mut visitor = inspector::ImguiVisitor::new(ui);
    let mut storage = ui.storage();
    let inspector = match storage.entry(String::from("inspectable_components")) {
        polymap::polymap::Entry::Vacant(entry) => {
            let mut i = inspector::Inspector::default();
            i.add::<crate::render::Shape>();
            i.add::<crate::render::light::Light>();
            i.add::<crate::math::Transform>();
            i.add::<crate::camera::Camera>();
            i.add::<crate::camera::FreeFlyCameraState>();
            i.add::<crate::common::Name>();
            entry.insert(i)
        }
        polymap::polymap::Entry::Occupied(entry) => entry.into_mut(),
    };

    inspector.inspect_components(&mut visitor, world, ent);
}

struct SelectedEntity {
    entity: specs::Entity,
}

#[derive(Default)]
pub struct EditorUiModule {}

use crate::render::ui::{UIModule, UiFrame};

impl UIModule for EditorUiModule {
    fn draw(&mut self, world: &mut World, frame: &UiFrame) {
        let dt = world.read_resource::<crate::time::Time>().delta_sim();
        let size = [400.0, 300.0];
        let pos = [0.0, 0.0];
        imgui::Window::new("Overview")
            .size(size, imgui::Condition::FirstUseEver)
            .position(pos, imgui::Condition::FirstUseEver)
            .build(frame.inner(), || {
                frame.inner().text(format!("FPS: {:.3}", dt.as_fps()));
                let mut p = crate::render::camera_pos(world).into_array();

                InputFloat3::new(frame.inner(), "Camera pos", &mut p)
                    .read_only(true)
                    .build();

                frame.inner().text("Right handed coordinate system");
            });

        {
            let mut y_offset = 0.0;
            let funcs = [crate::render::debug::window::build_ui];
            for func in funcs.iter() {
                let size = func(world, frame, [0.0, y_offset]);
                y_offset += size[1];
            }
        }

        let [width, _height] = frame.inner().io().display_size;
        let scene_window_size = [300.0, 500.0];
        let scene_window_pos = [width - scene_window_size[0], 0.0];

        let mut inspected: Option<specs::Entity> = None;

        {
            let parent_storage = world.read_storage::<graph::Parent>();
            let entities = world.read_resource::<specs::world::EntitiesRes>();

            imgui::Window::new("Scene")
                .position(scene_window_pos, Condition::Always)
                .size(scene_window_size, Condition::Always)
                .build(frame.inner(), || {
                    for (ent, _root) in (&entities, !&parent_storage).join() {
                        inspected = inspected.or_else(|| build_tree(world, frame, ent));
                    }
                });

            if world.has_value::<SelectedEntity>() && inspected.is_none() {
                inspected = Some(world.read_resource::<SelectedEntity>().entity);
            }
        }

        let inspected_window_size = [scene_window_size[0], 300.0];
        let inspected_window_pos = [scene_window_pos[0], scene_window_size[1]];
        if let Some(ent) = inspected {
            imgui::Window::new("Inspector")
                .position(inspected_window_pos, Condition::FirstUseEver)
                .size(inspected_window_size, Condition::FirstUseEver)
                .build(frame.inner(), || {
                    build_inspector(world, frame, ent);
                });
            world.insert(SelectedEntity { entity: ent });
        }
    }
}

pub fn ui_module() -> Box<dyn UIModule> {
    Box::new(EditorUiModule::default())
}
