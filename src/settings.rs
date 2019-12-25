use crate::input::{ActionId, InputContext, MappedInput};
use winit::VirtualKeyCode;

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
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Opaque,
        }
    }
}

fn get_input_context() -> InputContext {
    InputContext::start("RenderSettingsSys")
        .with_description("Input for changing render settings")
        .with_action(VirtualKeyCode::O, RENDER_MODE_SWITCH)
        .expect("Should only be one action!")
        .build()
}

const RENDER_MODE_SWITCH: ActionId = ActionId(0);

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
        log::debug!("RenderSettingsSys: run");
        for (inp, _id) in (&mut inputs, &unique_component).join() {
            if inp.contains_action(RENDER_MODE_SWITCH) {
                log::debug!("Render mode switch!");
                r_settings.render_mode = match r_settings.render_mode {
                    RenderMode::Opaque => RenderMode::Wireframe,
                    RenderMode::Wireframe => RenderMode::Opaque,
                };
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