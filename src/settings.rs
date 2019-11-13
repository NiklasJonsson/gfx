use crate::input::{ActionId, InputContext, MappedInput};
use winit::VirtualKeyCode;

use specs::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive)]
enum RenderMode {
    Opaque,
    Wireframe,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct RenderSettings {
    // Affects all entities
    rendering_mode: RenderMode,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            rendering_mode: RenderMode::Opaque,
        }
    }
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
        log::trace!("RenderSettingsSys: run");
        for (inp, _) in (&mut inputs, &unique_component).join() {
            if inp.contains_action(RENDER_MODE_SWITCH) {
                r_settings.rendering_mode = match r_settings.rendering_mode {
                    RenderMode::Opaque => RenderMode::Wireframe,
                    RenderMode::Wireframe => RenderMode::Opaque,
                };
            }
            inp.clear();
        }
    }
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        RenderSettingsSys,
        "rendering_settings_sys",
        &[crate::input::INPUT_MANAGER_SYSTEM_ID],
    )
}

fn get_input_context() -> InputContext {
    InputContext::start("RenderSettingsSys")
        .with_description("Input for changing render settings")
        .with_action(VirtualKeyCode::O, RENDER_MODE_SWITCH)
        .expect("Should only be one action!")
        .build()
}

pub fn init(world: &mut World) {
    world.insert(RenderSettings::default());
    let ctx = get_input_context();
    let mi = MappedInput::new();
    world
        .create_entity()
        .with(ctx)
        .with(mi)
        .with(RenderSettingsSys {})
        .build();
}
