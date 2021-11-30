use specs::prelude::*;

use crate::common::Name;
use crate::ecs;
use crate::graph;
use imgui::{im_str, CollapsingHeader, Condition, InputFloat3, TreeNode};
pub type Ui<'a> = crate::render::ui::UiFrame<'a>;

use crate::visit::{Meta, Visitor as _};

use ramneryd_derive::Visitable;

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

    let name = im_str!("{}", name(world, ent));
    TreeNode::new(&name).build(ui.inner(), || {
        let pressed = ui.inner().small_button(im_str!("inspect"));
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

    ui.inner().text(im_str!("{}", name(world, ent)));
    ui.inner().separator();
    let pressed = ui.inner().small_button(im_str!("reload material"));
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
            let mut i = inspector::Inspector::new();
            i.add::<crate::render::Shape>(crate::render::Shape::meta().name);
            i.add::<crate::render::light::Light>(crate::render::light::Light::meta().name);
            i.add::<crate::math::Transform>(crate::math::Transform::meta().name);
            i.add::<crate::camera::Camera>(crate::camera::Camera::meta().name);
            i.add::<crate::camera::FreeFlyCameraState>(
                crate::camera::FreeFlyCameraState::meta().name,
            );
            i.add::<crate::common::Name>(crate::common::Name::meta().name);
            //i.add::<crate::io::input::InputContext>(crate::io::input::InputContext::meta().name);
            entry.insert(i)
        }
        polymap::polymap::Entry::Occupied(entry) => entry.into_mut(),
    };

    for comp in ecs::meta::ALL_COMPONENTS {
        if (comp.has)(world, ent) {
            if comp.size == 0 {
                let _open = CollapsingHeader::new(&imgui::ImString::from(String::from(comp.name)))
                    .leaf(true)
                    .build(ui.inner());
            } else if inspector.can_inspect(comp.name) {
                inspector.inspect(comp.name, &mut visitor, world, ent);
            } else if CollapsingHeader::new(&imgui::ImString::from(format!("struct {}", comp.name)))
                .build(ui.inner())
            {
                ui.inner().text(im_str!("unimplemented"));
            }
        }
    }
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
        imgui::Window::new(im_str!("Overview"))
            .size(size, imgui::Condition::FirstUseEver)
            .position(pos, imgui::Condition::FirstUseEver)
            .build(frame.inner(), || {
                frame.inner().text(im_str!("FPS: {:.3}", dt.as_fps()));
                let mut p = crate::render::camera_pos(world).into_array();

                InputFloat3::new(frame.inner(), im_str!("Camera pos"), &mut p)
                    .read_only(true)
                    .build();

                let ffcs_storage = world.read_storage::<crate::camera::FreeFlyCameraState>();
                for (i, state) in ffcs_storage.join().enumerate() {
                    let ori = state.orientation();
                    let mut view_dir = ori.view_direction.into_array();
                    let mut up = ori.up.into_array();
                    InputFloat3::new(frame.inner(), &im_str!("view dir {}", i), &mut view_dir)
                        .read_only(true)
                        .build();

                    InputFloat3::new(frame.inner(), &im_str!("up {}", i), &mut up)
                        .read_only(true)
                        .build();
                }

                frame
                    .inner()
                    .text(im_str!("#components: {}", ecs::meta::ALL_COMPONENTS.len()));
                frame
                    .inner()
                    .text(im_str!("Right handed coordinate system"));
                frame.inner().text(im_str!("Registered systems:"));
            });

        {
            let mut y_offset = 0.0;
            let funcs = [
                crate::render::debug_window::build_ui,
                crate::io::input::build_ui,
            ];
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

            imgui::Window::new(im_str!("Scene"))
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
            imgui::Window::new(im_str!("Inspector"))
                .position(inspected_window_pos, Condition::FirstUseEver)
                .size(inspected_window_size, Condition::FirstUseEver)
                .build(frame.inner(), || {
                    build_inspector(world, frame, ent);
                });
            world.insert(SelectedEntity { entity: ent });
        }

        let mut s = frame.storage();
        let test = match s.entry(String::from("a struct")) {
            polymap::polymap::Entry::Vacant(entry) => {
                let a = A { x: 10, y: 40.0 };
                let s = S { u: 123123123 };
                let es = [
                    E::InlineStruct { s: 23478989 },
                    E::Struct(s),
                    E::TupleStruct(s, a, 6758.0, false),
                    E::Bare,
                ];
                let c = C {
                    e: E::TupleStruct(s, a, 6758324.0, true),
                };
                let b = B { a, s, es, c };
                entry.insert(b)
            }
            polymap::polymap::Entry::Occupied(entry) => entry.into_mut(),
        };
        let mut ins = inspector::ImguiVisitor::new(frame);
        let test_id = im_str!("Test reflection code");
        imgui::Window::new(test_id)
            .position(inspected_window_pos, Condition::FirstUseEver)
            .size(inspected_window_size, Condition::FirstUseEver)
            .build(frame.inner(), || {
                let t = frame.inner().push_id(test_id);
                ins.visit_mut(test, &Meta::standalone());
                t.pop(frame.inner());
            });
        let test_id = im_str!("Manual reflection code");
        imgui::Window::new(test_id)
            .position(inspected_window_pos, Condition::FirstUseEver)
            .size(inspected_window_size, Condition::FirstUseEver)
            .build(frame.inner(), || {
                let t = frame.inner().push_id(test_id);
                build_manual_test(test, frame);
                t.pop(frame.inner());
            });
    }
}

#[derive(Visitable, Clone, Copy)]
struct A {
    x: i32,
    y: f32,
}

#[derive(Visitable, Clone, Copy)]
struct S {
    u: usize,
}

#[derive(Visitable, Clone, Copy)]
enum E {
    InlineStruct { s: i32 },
    Struct(S),
    TupleStruct(S, A, f32, bool),
    Bare,
}

#[derive(Visitable, Clone, Copy)]
struct C {
    e: E,
}

#[derive(Visitable)]
struct B {
    a: A,
    s: S,
    c: C,
    es: [E; 4],
}

fn mk_field<'a>(ui: &imgui::Ui<'a>, s: &str) {
    ui.text(s);
    ui.same_line(0.0);
}

fn variant_name(e: &E) -> &str {
    ["InlineStruct", "Struct", "TupleStruct", "Base"][variant_idx(e)]
}

fn variant_idx(e: &E) -> usize {
    match e {
        E::InlineStruct { .. } => 0,
        E::Struct(..) => 1,
        E::TupleStruct(..) => 2,
        E::Bare => 3,
    }
}

fn inspect_struct<'a, Body>(name: &str, ty: &str, ui: &Ui<'a>, body: Option<Body>)
where
    Body: FnMut(),
{
    if !name.is_empty() {
        ui.inner().text(name);
        ui.inner().same_line(0.0);
    }

    let id = if name.is_empty() {
        im_str!("struct {}", ty)
    } else {
        im_str!("struct {}##{}", ty, name)
    };
    let token = ui.inner().push_id(&id);
    if imgui::CollapsingHeader::new(&id)
        .leaf(body.is_none())
        .build(ui.inner())
    {
        ui.inner().indent();
        if let Some(mut body) = body {
            body();
        }
        ui.inner().unindent();
    }
    token.pop(ui.inner());
}

fn build_manual_test(test: &mut B, frame: &UiFrame) {
    let header_label = im_str!("B");
    let token = frame.inner().push_id(header_label);

    let ui = frame.inner();
    if imgui::CollapsingHeader::new(header_label).build(ui) {
        ui.indent();
        inspect_struct(
            "a",
            "A",
            frame,
            Some(|| {
                ui.input_int(im_str!("x"), &mut test.a.x).build();
                ui.input_float(im_str!("y"), &mut test.a.y).build();
            }),
        );
        inspect_struct(
            "s",
            "S",
            frame,
            Some(|| {
                ui.text(&format!("u: {}", test.s.u));
            }),
        );

        inspect_struct(
            "c",
            "C",
            frame,
            Some(|| {
                mk_field(ui, "e");
                if imgui::CollapsingHeader::new(&im_str!("E::{}", variant_name(&test.c.e)))
                    .build(ui)
                {
                    ui.indent();
                    match test.c.e {
                        E::Bare => (),
                        E::Struct(s) => {
                            inspect_struct(
                                "",
                                "S",
                                frame,
                                Some(|| {
                                    ui.text(&format!("u: {}", s.u));
                                }),
                            );
                        }
                        E::InlineStruct { mut s } => {
                            ui.input_int(im_str!("s"), &mut s).build();
                        }
                        E::TupleStruct(s, mut a, mut f, mut b) => {
                            inspect_struct(
                                "",
                                "S",
                                frame,
                                Some(|| {
                                    ui.text(&format!("u: {}", s.u));
                                }),
                            );

                            inspect_struct(
                                "",
                                "A",
                                frame,
                                Some(|| {
                                    ui.input_int(im_str!("x"), &mut a.x).build();
                                    ui.input_float(im_str!("y"), &mut a.y).build();
                                }),
                            );

                            ui.input_float(im_str!(""), &mut f).build();
                            ui.checkbox(im_str!(""), &mut b);
                        }
                    }
                    ui.unindent();
                }
            }),
        );
        ui.unindent();
    }
    token.pop(frame.inner());
}

pub fn ui_module() -> Box<dyn UIModule> {
    Box::new(EditorUiModule::default())
}
