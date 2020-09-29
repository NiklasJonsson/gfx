use crate::io::input::{ActionId, InputContext, MappedInput};
use specs::Component;
use winit::event::VirtualKeyCode;

use num_derive::FromPrimitive;

use specs::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive)]
pub enum RenderMode {
    Opaque,
    Wireframe,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RenderSettings {
    // Affects all entities
    pub render_mode: RenderMode,
    pub render_bounding_box: bool,
    pub reload_runtime_shaders: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Opaque,
            render_bounding_box: false,
            reload_runtime_shaders: false,
        }
    }
}

fn get_input_context() -> InputContext {
    InputContext::start("RenderSettingsSys")
        .with_description("Input for changing render settings")
        .with_action(VirtualKeyCode::O, RENDER_MODE_SWITCH)
        .unwrap()
        .with_action(VirtualKeyCode::P, RENDER_BOUNDING_BOX_SWITCH)
        .unwrap()
        .with_action(VirtualKeyCode::R, RELOAD_RUNTIME_SHADERS)
        .unwrap()
        .build()
}

const RENDER_MODE_SWITCH: ActionId = ActionId(0);
const RENDER_BOUNDING_BOX_SWITCH: ActionId = ActionId(1);
const RELOAD_RUNTIME_SHADERS: ActionId = ActionId(2);

#[derive(Default, Component)]
#[storage(NullStorage)]
struct RenderSettingsSys;

// Use NullStorage and ReadStorage<'a, Self> to only access this system's MappedInput and
// InputContext.
impl<'a> System<'a> for RenderSettingsSys {
    type SystemData = (
        Write<'a, RenderSettings>,
        WriteStorage<'a, MappedInput>,
        ReadStorage<'a, Self>,
    );

    fn run(&mut self, (mut r_settings, mut inputs, unique_component): Self::SystemData) {
        log::trace!("RenderSettingsSys: run");
        for (inp, _id) in (&mut inputs, &unique_component).join() {
            if inp.contains_action(RENDER_MODE_SWITCH) {
                log::debug!("Render mode switch!");
                r_settings.render_mode = match r_settings.render_mode {
                    RenderMode::Opaque => RenderMode::Wireframe,
                    RenderMode::Wireframe => RenderMode::Opaque,
                };
            }
            if inp.contains_action(RENDER_BOUNDING_BOX_SWITCH) {
                log::debug!("Render bounding box switch!");
                r_settings.render_bounding_box = !r_settings.render_bounding_box;
            }

            if inp.contains_action(RELOAD_RUNTIME_SHADERS) {
                log::debug!("Reload runtime shaders!");
                r_settings.reload_runtime_shaders = true;
            }

            inp.clear();
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        world.insert(RenderSettings::default());
        let ctx = get_input_context();
        let mi = MappedInput::new();
        world
            .create_entity()
            .with(ctx)
            .with(mi)
            .with(RenderSettingsSys)
            .build();
    }
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(RenderSettingsSys, "rendering_settings_sys", &[])
}

pub fn build_ui<'a>(world: &World, ui: &imgui::Ui<'a>, pos: [f32; 2]) -> [f32; 2] {
    let settings = world.read_resource::<RenderSettings>();

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
                settings.reload_runtime_shaders
            ));
        });

    size
}
