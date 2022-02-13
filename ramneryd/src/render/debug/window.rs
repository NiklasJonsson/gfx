use crate::camera::FreeFlyCameraState;
use crate::common::Name;
use crate::ecs::prelude::*;
use crate::io::input::{ActionId, InputContext, InputContextError, KeyCode, MappedInput};
use crate::math::{Rgb, Transform, Vec3};
use crate::render;
use crate::visit::{Meta, MetaOrigin, Visitor};
use render::Light;

use crate::editor::inspector::ImguiVisitor;
use crate::visit::Visitable as _;
use render::ui::UiFrame;

use ramneryd_derive::Visitable;

use num_derive::FromPrimitive;

use std::borrow::Cow;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive, Visitable)]
pub enum RenderMode {
    Opaque,
    Wireframe,
}

#[derive(Default)]
struct AddLightModalState {
    idx: usize,
    choice: Light,
    name: Name,
    tfm: Transform,
}

#[derive(Default)]
struct RenderSettingsState {
    add_light_modal: Option<AddLightModalState>,
}

#[derive(Visitable)]
pub struct RenderSettings {
    // Affects all entities
    pub render_mode: RenderMode,
    pub render_bounding_box: bool,
    pub reload_shaders: bool,
    pub render_light_volumes: bool,

    // TODO: Move to other state holder
    #[visitable(ignore)]
    state: RenderSettingsState,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Opaque,
            render_bounding_box: false,
            reload_shaders: false,
            render_light_volumes: false,
            state: RenderSettingsState::default(),
        }
    }
}

fn get_input_context() -> Result<InputContext, InputContextError> {
    Ok(InputContext::builder(RenderSettingsSys::ID)
        .description("Input for changing render settings")
        .with_action(KeyCode::O, RENDER_MODE_SWITCH)?
        .with_action(KeyCode::P, RENDER_BOUNDING_BOX_SWITCH)?
        .with_action(KeyCode::R, RELOAD_SHADERS)?
        .build())
}

const RENDER_MODE_SWITCH: ActionId = ActionId(0);
const RENDER_BOUNDING_BOX_SWITCH: ActionId = ActionId(1);
const RELOAD_SHADERS: ActionId = ActionId(2);

struct RenderSettingsSys {
    input_entity: Option<specs::Entity>,
}

impl RenderSettingsSys {
    pub const ID: &'static str = "RenderSettingsSys";
}

impl<'a> System<'a> for RenderSettingsSys {
    type SystemData = (Write<'a, RenderSettings>, WriteStorage<'a, MappedInput>);

    fn run(&mut self, (mut r_settings, mut inputs): Self::SystemData) {
        use crate::io::input::Input;
        log::trace!("RenderSettingsSys: run");
        let inp = inputs
            .get_mut(self.input_entity.unwrap())
            .expect("Failed to get mapped input for RenderSettingsSys");

        for i in inp.iter() {
            match i {
                Input::Action(RENDER_MODE_SWITCH) => {
                    log::debug!("Render mode switch!");
                    r_settings.render_mode = match r_settings.render_mode {
                        RenderMode::Opaque => RenderMode::Wireframe,
                        RenderMode::Wireframe => RenderMode::Opaque,
                    };
                }
                Input::Action(RENDER_BOUNDING_BOX_SWITCH) => {
                    log::debug!("Render bounding box switch!");
                    r_settings.render_bounding_box = !r_settings.render_bounding_box;
                }
                Input::Action(RELOAD_SHADERS) => {
                    log::debug!("Reload runtime shaders!");
                    r_settings.reload_shaders = true;
                }
                i => unreachable!("{:?}", i),
            }
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        world.insert(RenderSettings::default());
        let ctx = get_input_context().expect("Failed to build settings input context");
        self.input_entity = Some(
            world
                .create_entity()
                .with(ctx)
                .with(Name::from(RenderSettingsSys::ID))
                .build(),
        );
    }
}

#[derive(Visitable)]
struct LightInfo {
    transform: Transform,
    light: Light,
}

fn build_lights_tab(
    world: &mut World,
    visitor: &mut ImguiVisitor,
    frame: &crate::render::ui::UiFrame,
) {
    {
        let mut lights = world.write_storage::<Light>();
        let mut transforms = world.write_storage::<crate::math::Transform>();
        for (i, (light, tfm)) in (&mut lights, &mut transforms).join().enumerate() {
            let mut l = LightInfo {
                transform: *tfm,
                light: light.clone(),
            };

            let idx: u8 = i.try_into().expect("Too many lights");

            visitor.visit_mut(
                &mut l,
                &Meta {
                    type_name: "Light",
                    range: None,
                    origin: MetaOrigin::TupleField { idx },
                },
            );
        }
    }

    {
        let modal_id = "New light";
        if frame.inner().button("Add") {
            frame.inner().open_popup(modal_id);
        }
        frame
            .inner()
            .popup_modal(modal_id)
            .build(frame.inner(), || {
                let mut modal_state = world
                    .write_resource::<RenderSettings>()
                    .state
                    .add_light_modal
                    .take();
                // TODO(refactor): Auto-generate some of this
                let items = [("Point"), ("Directional"), ("Spot"), ("Ambient")];
                let mut idx = modal_state.as_ref().map(|x| x.idx).unwrap_or(0);
                let selected = frame.inner().combo("Selected", &mut idx, &items, |s| {
                    std::borrow::Cow::Borrowed(*s)
                });
                if selected || modal_state.is_none() {
                    let tfm = Transform::default();
                    let name = Name::from(items[idx].to_string());
                    let light = match idx {
                        0 => Light::Point {
                            color: Rgb {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                            },
                            range: 5.0,
                        },
                        1 => Light::Directional {
                            color: Rgb {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                            },
                        },
                        2 => Light::Spot {
                            color: Rgb {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                            },
                            angle: std::f32::consts::FRAC_PI_8,
                            range: std::ops::Range {
                                start: 0.1,
                                end: 5.0,
                            },
                        },
                        3 => Light::Ambient {
                            color: Rgb {
                                r: 1.0,
                                g: 1.0,
                                b: 1.0,
                            },
                            strength: 0.05,
                        },
                        x => unreachable!("Invalid value for light selection {}", x),
                    };
                    modal_state = Some(AddLightModalState {
                        choice: light,
                        idx,
                        tfm,
                        name,
                    });
                }
                let mut create = false;
                if let Some(AddLightModalState {
                    choice, name, tfm, ..
                }) = &mut modal_state
                {
                    visitor.visit_mut(choice, &Meta::field("properties"));
                    visitor.visit_mut(tfm, &Meta::field("transform"));
                    visitor.visit_mut(name, &Meta::field("name"));

                    if frame.inner().button("Create") {
                        create = true;
                        frame.inner().close_current_popup();
                    }
                    frame.inner().same_line();
                    if frame.inner().button("Cancel") {
                        create = false;
                        frame.inner().close_current_popup();
                    }
                }
                if create {
                    let AddLightModalState {
                        choice, name, tfm, ..
                    } = modal_state.expect("create should only be set if this existed");
                    world
                        .create_entity()
                        .with(choice)
                        .with(tfm)
                        .with(name)
                        .build();
                } else {
                    world
                        .write_resource::<RenderSettings>()
                        .state
                        .add_light_modal = modal_state;
                }
            });
    }
}

fn build_overview_tab(world: &mut World, visitor: &mut ImguiVisitor, _frame: &UiFrame) {
    let mut settings = world.write_resource::<RenderSettings>();
    settings.visit_fields_mut(visitor);
}

#[derive(Debug, Visitable)]
struct CameraInfo {
    pos: Vec3,
    view_dir: Vec3,
    shadow_viewer: bool,
    main_render_camera: bool,
    entity: Entity,
    draw_frustum: bool,
}

struct MainCameraData {
    ent: Entity,
    idx: usize,
    tfm: crate::math::Transform,
    cam: crate::camera::Camera,
    ff_state: crate::camera::FreeFlyCameraState,
}

fn build_cameras_tab(world: &mut World, visitor: &mut ImguiVisitor, frame: &UiFrame) {
    use crate::render::debug::DrawFrustum;
    use crate::render::light::ShadowViewer;
    use crate::render::{Camera, MainRenderCamera};

    type SysData<'a> = (
        Entities<'a>,
        ReadStorage<'a, Camera>,
        ReadStorage<'a, Transform>,
        ReadStorage<'a, FreeFlyCameraState>,
        ReadStorage<'a, ShadowViewer>,
        ReadStorage<'a, MainRenderCamera>,
        WriteStorage<'a, DrawFrustum>,
    );

    // TODO (perf): tmp alloc here
    let mut camera_entities = Vec::new();
    let mut main_camera_data = None;
    {
        let (
            entities,
            cameras,
            transforms,
            states,
            shadow_viewers,
            main_render_cameras,
            mut draw_frustum_tags,
        ) = SysData::fetch(world);
        for (i, (ent, cam, tfm, state, shadow_viewer, main_cam)) in (
            &entities,
            &cameras,
            &transforms,
            &states,
            shadow_viewers.maybe(),
            main_render_cameras.maybe(),
        )
            .join()
            .enumerate()
        {
            camera_entities.push((ent, i));
            if main_cam.is_some() && main_camera_data.is_none() {
                main_camera_data = Some(MainCameraData {
                    ent,
                    idx: i,
                    tfm: *tfm,
                    cam: *cam,
                    ff_state: *state,
                });
            }

            let has_draw_frustum = draw_frustum_tags.get(ent).is_some();
            let mut c = CameraInfo {
                pos: tfm.position,
                view_dir: Camera::view_direction(tfm),
                shadow_viewer: shadow_viewer.is_some(),
                main_render_camera: main_cam.is_some(),
                entity: ent,
                draw_frustum: has_draw_frustum,
            };

            let idx: u8 = i.try_into().expect("Too many lights");
            visitor.visit_mut(
                &mut c,
                &Meta {
                    type_name: "Camera",
                    range: None,
                    origin: MetaOrigin::TupleField { idx },
                },
            );

            if c.draw_frustum && !has_draw_frustum {
                // User checked the box
                draw_frustum_tags
                    .insert(ent, DrawFrustum)
                    .expect("Failed to add DrawFrustum");
            }

            if !c.draw_frustum && has_draw_frustum {
                // User unchecked the box
                draw_frustum_tags
                    .remove(ent)
                    .expect("Failed to remove DrawFrustum");
            }
        }
    }

    let mut main_camera_data = if let Some(data) = main_camera_data {
        data
    } else {
        log::error!(
            "Tried to render camera tab for render debug ui but failed to find main camera"
        );
        return;
    };

    {
        let ui = frame.inner();
        if ui.button("Add") {
            let mut tfm = main_camera_data.tfm;
            tfm.position += Vec3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            };
            world
                .create_entity()
                .with(main_camera_data.cam)
                .with(tfm)
                .with(main_camera_data.ff_state)
                .with(Name::from(format!("{}", camera_entities.len())))
                .build();
        }

        if ui.combo(
            "Main render camera",
            &mut main_camera_data.idx,
            &camera_entities,
            |e| Cow::Owned(format!("{}", e.1)),
        ) {
            let old_main = main_camera_data.ent;
            let new_main = camera_entities[main_camera_data.idx].0;
            if old_main != new_main {
                let mut marker_storage = world.write_storage::<MainRenderCamera>();
                marker_storage
                    .remove(old_main)
                    .expect("Main camera should have this marker");

                let mut input_ctx_storage = world.write_storage::<InputContext>();
                let input_ctx = input_ctx_storage
                    .remove(old_main)
                    .expect("Main camera should have input context");

                let old = marker_storage
                    .insert(new_main, MainRenderCamera)
                    .expect("Failed to switch main render camera");
                assert!(
                    old.is_none(),
                    "New main render camera should not have had the tag already"
                );
                let old = input_ctx_storage
                    .insert(new_main, input_ctx)
                    .expect("Failed to move input context to new main camera");
                assert!(old.is_none(), "New main camera already has a context!?");
            }
        }
    }
}

pub(crate) fn build_ui<'a>(
    world: &mut World,
    ui: &crate::render::ui::UiFrame<'a>,
    pos: [f32; 2],
) -> [f32; 2] {
    type TabItemFn = fn(&mut World, &mut ImguiVisitor, &UiFrame);

    let size = [300.0, 85.0];

    let mut visitor = crate::editor::inspector::ImguiVisitor::new(ui);
    imgui::Window::new("Render debug")
        .position(pos, imgui::Condition::FirstUseEver)
        .size(size, imgui::Condition::FirstUseEver)
        .build(ui.inner(), || {
            let inner = ui.inner();
            if let Some(token) = imgui::TabBar::new("RenderDebugTabBar)").begin(inner) {
                let items = [
                    ("Overview", build_overview_tab as TabItemFn),
                    ("Lights", build_lights_tab as TabItemFn),
                    ("Cameras", build_cameras_tab as TabItemFn),
                ];

                for (id, func) in items {
                    imgui::TabItem::new(id).build(inner, || func(world, &mut visitor, ui));
                }
                token.end();
            }
        });

    size
}

pub struct ApplySettings;

impl ApplySettings {
    pub const ID: &'static str = "ApplySettings";
}

impl<'a> System<'a> for ApplySettings {
    type SystemData = (
        Write<'a, RenderSettings>,
        Entities<'a>,
        ReadStorage<'a, render::material::GpuMaterial>,
        WriteStorage<'a, render::ReloadMaterial>,
        ReadStorage<'a, crate::math::Aabb>,
        WriteStorage<'a, super::bounding_box::RenderBoundingBox>,
        ReadStorage<'a, Light>,
        WriteStorage<'a, super::light::RenderLightVolume>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (
            mut render_settings,
            entities,
            materials,
            mut reload_materials,
            bounding_boxes,
            mut render_bbox,
            lights,
            mut render_light_cmds,
        ) = data;
        if render_settings.reload_shaders {
            for (ent, _mat) in (&entities, &materials).join() {
                reload_materials
                    .insert(ent, render::ReloadMaterial)
                    .expect("Failed to insert");
            }

            render_settings.reload_shaders = false;
        }

        if render_settings.render_bounding_box {
            for (ent, _bbox) in (&entities, &bounding_boxes).join() {
                if render_bbox.get(ent).is_none() {
                    render_bbox
                        .insert(ent, super::bounding_box::RenderBoundingBox)
                        .expect("Failed to insert");
                }
            }
        }

        if render_settings.render_light_volumes {
            for (ent, _light) in (&entities, &lights).join() {
                if render_light_cmds.get(ent).is_none() {
                    render_light_cmds
                        .insert(ent, super::light::RenderLightVolume)
                        .expect("Failed to insert");
                }
            }
        } else {
            render_light_cmds.clear();
        }
    }
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    builder
        .with(
            RenderSettingsSys { input_entity: None },
            RenderSettingsSys::ID,
            &[],
        )
        .with(ApplySettings, ApplySettings::ID, &[RenderSettingsSys::ID])
}
