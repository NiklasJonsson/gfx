use std::collections::HashSet;

use ecs::prelude::*;

use crate::camera::Camera;
use crate::common::Name;
use crate::render::imgui::{UIModule, UiFrame};
use crate::{ecs, graph};
use imgui::{Condition, InputFloat3};

pub mod inspector;

fn name(world: &World, ent: Entity) -> String {
    let ent_id = ent.id();
    let names = world.read_storage::<Name>();
    match names.get(ent) {
        Some(name) => format!("{ent_id} {}", name.0),
        None => format!("{ent_id}"),
    }
}

fn build_tree(
    world: &World,
    ui: &crate::render::imgui::UiFrame<'_>,
    ent: specs::Entity,
) -> Option<specs::Entity> {
    let mut inspected = None;
    let has_children = world.has_component::<graph::Children>(ent);
    let tree_node = ui
        .inner()
        .tree_node_config(name(world, ent))
        .leaf(!has_children);

    let mut inspect_button = || {
        ui.inner().same_line();
        let pressed = ui
            .inner()
            .small_button(std::format!("inspect##{}", ent.id()));
        if pressed {
            inspected = Some(ent);
        }
    };

    if let Some(_token) = tree_node.push() {
        inspect_button();
        if let Some(children) = world.read_storage::<graph::Children>().get(ent) {
            for child in children.iter() {
                let new = build_tree(world, ui, *child);
                inspected = inspected.or(new);
            }
        }
    } else {
        inspect_button();
    }

    inspected
}

fn build_inspector(world: &mut World, ui: &crate::render::imgui::UiFrame<'_>, ent: specs::Entity) {
    use crate::render::ReloadMaterial;
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

#[derive(Default)]
struct SelectedEntity {
    entities: HashSet<ecs::Entity>,
}

#[derive(Default)]
pub struct EditorUiModule {}

impl UIModule for EditorUiModule {
    fn draw(&mut self, world: &mut World, frame: &UiFrame) {
        let dt = world.read_resource::<crate::time::Time>().delta_sim();
        let size = [400.0, 300.0];
        let pos = [0.0, 0.0];
        frame
            .inner()
            .window("Overview")
            .size(size, imgui::Condition::FirstUseEver)
            .position(pos, imgui::Condition::FirstUseEver)
            .build(|| {
                let ui = frame.inner();
                ui.text(format!("FPS: {:.3}", dt.as_fps()));

                if ui.collapsing_header("Camera", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    let camera_entity =
                        ecs::get_singleton_entity::<crate::render::MainRenderCamera>(world);
                    let mut tfm_storage = world.write_storage::<crate::math::Transform>();
                    let tfm = tfm_storage
                        .get_mut(camera_entity)
                        .expect("No transform for camera");

                    {
                        let view_dir = Camera::view_direction(tfm);
                        let mut arr = view_dir.into_array();
                        InputFloat3::new(frame.inner(), "View direction", &mut arr)
                            .read_only(true)
                            .build();
                    }
                    {
                        let pos = &mut tfm.position;
                        let mut arr = pos.into_array();
                        if InputFloat3::new(frame.inner(), "Position", &mut arr).build() {
                            *pos = arr.into();
                        }
                    }

                    {
                        let mut cams = world.write_storage::<Camera>();
                        let cam = cams.get_mut(camera_entity).unwrap();
                        inspector::draw_struct_mut(frame, "Render camera", cam)
                    }
                }

                frame.inner().text("Right handed coordinate system");
            });

        let [width, _height] = frame.inner().io().display_size;
        let scene_window_size = [300.0, 500.0];
        let scene_window_pos = [width - scene_window_size[0], 0.0];

        let mut inspected: Option<specs::Entity> = None;

        {
            let parent_storage = world.read_storage::<graph::Parent>();
            let entities = world.read_resource::<specs::world::EntitiesRes>();

            frame
                .inner()
                .window("Entities")
                .position(scene_window_pos, Condition::Always)
                .size(scene_window_size, Condition::Always)
                .build(|| {
                    for (ent, _root) in (&entities, !&parent_storage).join() {
                        let inspect = build_tree(world, frame, ent);
                        inspected = inspected.or(inspect);
                    }
                });
        }

        let mut selected = world.remove::<SelectedEntity>().unwrap_or_default();

        if let Some(ent) = inspected {
            selected.entities.insert(ent);
        }

        let inspected_window_size = [scene_window_size[0], 300.0];
        let mut closed: Vec<ecs::Entity> = Vec::new();
        for ent in &selected.entities {
            let window_title = { std::format!("Entity {}", name(world, *ent)) };
            let mut opened = true;
            frame
                .inner()
                .window(window_title)
                .size(inspected_window_size, Condition::FirstUseEver)
                .opened(&mut opened)
                .build(|| {
                    build_inspector(world, frame, *ent);
                });
            if !opened {
                closed.push(*ent);
            }
        }
        for ent in closed {
            selected.entities.remove(&ent);
        }
        world.insert(selected);
    }
}

pub fn ui_module() -> Box<dyn UIModule> {
    Box::<EditorUiModule>::default()
}
