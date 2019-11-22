use crate::input::{ActionId, InputContext, InputContextPriority, MappedInput};

use specs::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GameState {
    Paused,
    Running,
}

impl Default for GameState {
    fn default() -> Self {
        GameState::Running
    }
}

const GAME_STATE_SWITCH: ActionId = ActionId(0);

#[derive(Default, Component)]
#[storage(NullStorage)]
struct GameStateSwitcher;

// Use NullStorage and ReadStorage<'a, Self> to only access this system's MappedInput and
// InputContext. Only one game object will have GameStateSwitcher as null storage, the one holding
// the right mapped input and input context.
impl<'a> System<'a> for GameStateSwitcher {
    type SystemData = (
        Write<'a, GameState>,
        WriteStorage<'a, InputContext>,
        WriteStorage<'a, MappedInput>,
        ReadStorage<'a, Self>,
    );

    fn run(&mut self, (mut state, mut contexts, mut inputs, unique_component): Self::SystemData) {
        log::trace!("GameStateSwitcher: run");
        for (inp, ctx, _) in (&mut inputs, &mut contexts, &unique_component).join() {
            use GameState::*;
            if inp.contains_action(GAME_STATE_SWITCH) {
                *state = match *state {
                    Paused => Running,
                    Running => Paused,
                };

                ctx.set_consume_all(*state == Paused);
            }
            inp.clear();
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        world.insert(GameState::default());
        let escape_catcher = InputContext::start("EscapeCatcher")
            .with_description("Global top-level escape catcher for game state switcher")
            .with_action(winit::VirtualKeyCode::Escape, GAME_STATE_SWITCH)
            .expect("Could not insert Escape action for GameStateSwitcher")
            .with_priority(InputContextPriority::First)
            .build();
        let mi = MappedInput::new();
        world
            .create_entity()
            .with(mi)
            .with(escape_catcher)
            .with(GameStateSwitcher {})
            .build();
    }
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        GameStateSwitcher,
        "game_state_switcher",
        &[crate::input::INPUT_MANAGER_SYSTEM_ID],
    )
}
