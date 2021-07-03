use crate::common::Name;
use crate::ecs::prelude::*;
use crate::io::input::{ActionId, InputContext, InputContextError, KeyCode, MappedInput};
use crate::math::{Rgb, Transform};
use crate::render;

use crate::editor;
use editor::Inspect as _;
use ramneryd_derive::Inspect;

use num_derive::FromPrimitive;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive, Inspect)]
pub enum RenderMode {
    Opaque,
    Wireframe,
}

#[derive(Default)]
struct AddLightModalState {
    idx: usize,
    choice: render::light::Light,
    name: Name,
    tfm: Transform,
}

#[derive(Default)]
struct RenderSettingsState {
    add_light_modal: Option<AddLightModalState>,
}

#[derive(Inspect)]
pub struct RenderSettings {
    // Affects all entities
    pub render_mode: RenderMode,
    pub render_bounding_box: bool,
    pub reload_shaders: bool,
    pub render_light_volumes: bool,

    #[inspect(ignore)]
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

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder
        .with(
            RenderSettingsSys { input_entity: None },
            RenderSettingsSys::ID,
            &[],
        )
        .with(ApplySettings, ApplySettings::ID, &[RenderSettingsSys::ID])
}

pub(crate) fn build_ui<'a>(
    world: &mut World,
    ui: &crate::render::ui::UiFrame<'a>,
    pos: [f32; 2],
) -> [f32; 2] {
    let size = [300.0, 85.0];

    imgui::Window::new(imgui::im_str!("Render debug"))
        .position(pos, imgui::Condition::FirstUseEver)
        .size(size, imgui::Condition::FirstUseEver)
        .build(ui.inner(), || {
            {
                let mut settings = world.write_resource::<RenderSettings>();
                settings.inspect_mut(ui, "");
                ui.inner().text("Lights");
                let mut lights = world.write_storage::<render::light::Light>();
                let mut transforms = world.write_storage::<crate::math::Transform>();
                for (i, (light, tfm)) in (&mut lights, &mut transforms).join().enumerate() {
                    let id = format!("Light##{}", i);
                    editor::inspect::inspect_struct(
                        "",
                        Some(&id),
                        ui,
                        Some(|| {
                            tfm.inspect_mut(ui, "transform");
                            light.inspect_mut(ui, "light");
                        }),
                    );
                }
            }

            {
                let modal_id = imgui::im_str!("New light");
                if ui.inner().button(imgui::im_str!("Add light"), [0.0, 0.0]) {
                    ui.inner().open_popup(modal_id);
                }
                ui.inner().popup_modal(modal_id).build(|| {
                    let mut modal_state = world
                        .write_resource::<RenderSettings>()
                        .state
                        .add_light_modal
                        .take();
                    // TODO(refactor): Auto-generate some of this
                    let items = [
                        imgui::im_str!("Point"),
                        imgui::im_str!("Directional"),
                        imgui::im_str!("Spot"),
                        imgui::im_str!("Ambient"),
                    ];
                    let mut idx = modal_state.as_ref().map(|x| x.idx).unwrap_or(0);
                    let selected = imgui::ComboBox::new(imgui::im_str!("")).build_simple_string(
                        ui.inner(),
                        &mut idx,
                        &items,
                    );
                    if selected || modal_state.is_none() {
                        let tfm = Transform::default();
                        let name = Name::from(items[idx].to_string());
                        let light = match idx {
                            0 => render::light::Light::Point {
                                color: Rgb {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                },
                                range: 5.0,
                            },
                            1 => render::light::Light::Directional {
                                color: Rgb {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                },
                            },
                            2 => render::light::Light::Spot {
                                color: Rgb {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                },
                                angle: std::f32::consts::FRAC_PI_8,
                                range: 5.0,
                            },
                            3 => render::light::Light::Ambient {
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
                        choice.inspect_mut(ui, "properties");
                        tfm.inspect_mut(ui, "transform");
                        name.inspect_mut(ui, "name");

                        if ui.inner().button(imgui::im_str!("Create"), [0.0, 0.0]) {
                            create = true;
                            ui.inner().close_current_popup();
                        }
                        ui.inner().same_line(0.0);
                        if ui.inner().button(imgui::im_str!("Cancel"), [0.0, 0.0]) {
                            create = false;
                            ui.inner().close_current_popup();
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
        ReadStorage<'a, crate::math::BoundingBox>,
        WriteStorage<'a, render::bounding_box::RenderBoundingBox>,
        ReadStorage<'a, render::light::Light>,
        WriteStorage<'a, render::light::RenderLightVolume>,
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
                        .insert(ent, render::bounding_box::RenderBoundingBox)
                        .expect("Failed to insert");
                }
            }
        }

        if render_settings.render_light_volumes {
            for (ent, _light) in (&entities, &lights).join() {
                if render_light_cmds.get(ent).is_none() {
                    render_light_cmds
                        .insert(ent, render::light::RenderLightVolume)
                        .expect("Failed to insert");
                }
            }
        } else {
            render_light_cmds.clear();
        }
    }
}
