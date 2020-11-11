use crate::common::Name;
use crate::io::input::{ActionId, InputContext, InputContextError, KeyCode, MappedInput};
use crate::math::Vec3;

use num_derive::FromPrimitive;

use specs::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive)]
pub enum RenderMode {
    Opaque,
    Wireframe,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RenderSettings {
    // Affects all entities
    pub render_mode: RenderMode,
    pub render_bounding_box: bool,
    pub reload_shaders: bool,
    pub light_pos: Vec3,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Opaque,
            render_bounding_box: false,
            reload_shaders: false,
            light_pos: Vec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
            },
        }
    }
}

const NAME: &str = "RenderSettingsSys";

fn get_input_context() -> Result<InputContext, InputContextError> {
    Ok(InputContext::builder(NAME)
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
                .with(Name::from(NAME))
                .build(),
        );
    }
}

pub const RENDER_SETTINGS_SYS_ID: &str = "render_settings_sys";

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        RenderSettingsSys { input_entity: None },
        RENDER_SETTINGS_SYS_ID,
        &[],
    )
}

pub fn build_ui<'a>(world: &mut World, ui: &imgui::Ui<'a>, pos: [f32; 2]) -> [f32; 2] {
    let mut settings = world.write_resource::<RenderSettings>();

    let size = [300.0, 85.0];

    imgui::Window::new(imgui::im_str!("Render settings"))
        .position(pos, imgui::Condition::FirstUseEver)
        .size(size, imgui::Condition::FirstUseEver)
        .build(&ui, || {
            ui.text(imgui::im_str!("render mode: {:?}", settings.render_mode));
            ui.text(imgui::im_str!(
                "bounding box: {}",
                settings.render_bounding_box
            ));
            ui.text(imgui::im_str!(
                "reload shaders: {}",
                settings.reload_shaders
            ));
            let mut pos = settings.light_pos.into_array();
            imgui::InputFloat3::new(ui, imgui::im_str!("Position"), &mut pos).build();
            settings.light_pos = Vec3::from(pos);
        });

    size
}
