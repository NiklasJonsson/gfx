use crate::io::input::{ActionId, InputContext, InputContextPriority, MappedInput};

use crate::io::input;

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

struct GameStateSwitcher {
    input_entity: Option<specs::Entity>,
}

impl<'a> System<'a> for GameStateSwitcher {
    type SystemData = (
        Write<'a, GameState>,
        WriteStorage<'a, InputContext>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (mut state, mut contexts, mut inputs): Self::SystemData) {
        log::trace!("GameStateSwitcher: run");

        let ent = self.input_entity.unwrap();
        let inp = inputs.get_mut(ent).unwrap();
        let ctx = contexts.get_mut(ent).unwrap();

        if inp.contains_action(GAME_STATE_SWITCH) {
            *state = match *state {
                GameState::Paused => GameState::Running,
                GameState::Running => GameState::Paused,
            };

            log::debug!("GameStateSwitcher: set state: {:?}", *state);

            ctx.set_consume_all(*state == GameState::Paused);
        }

        inp.clear();
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        world.insert(GameState::default());

        let escape_catcher = InputContext::start("EscapeCatcher")
            .with_description("Global top-level escape catcher for game state switcher")
            .with_action(input::KeyCode::Escape, GAME_STATE_SWITCH)
            .expect("Could not insert Escape action for GameStateSwitcher")
            .with_priority(InputContextPriority::First)
            .build();

        self.input_entity = Some(world.create_entity().with(escape_catcher).build());
    }
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        GameStateSwitcher { input_entity: None },
        "game_state_switcher",
        &[input::INPUT_MANAGER_SYSTEM_ID],
    )
}

pub fn build_ui<'a>(world: &World, ui: &imgui::Ui<'a>, pos: [f32; 2]) -> [f32; 2] {
    let state = world.read_resource::<GameState>();

    let size = [300.0, 50.0];

    imgui::Window::new(imgui::im_str!("Game state"))
        .position(pos, imgui::Condition::FirstUseEver)
        .size(size, imgui::Condition::FirstUseEver)
        .build(&ui, || {
            ui.text(imgui::im_str!("Game state: {:?}", *state));
        });

    size
}
