use crate::WindowEvents;
use log::{debug, info};
use specs::prelude::*;
use std::collections::HashMap;
use winit::{Event, VirtualKeyCode, WindowEvent};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975
//
// Basic idea:
// InputContext acts as a filter for inputs (key press/release, mouse click etc...). There are several InputContexts that consume input and if it's not interested in some input, it passes it on.

pub type ActionId = u32;
pub type ActionMap = HashMap<VirtualKeyCode, ActionId>;

/// A context that may consume some input and map it to an ActionId
// TODO: State, Range
#[derive(Component)]
#[storage(HashMapStorage)]
pub struct InputContext {
    name: String,
    description: String,
    action_map: ActionMap,
}

impl InputContext {
    pub fn new(name: &str, description: &str, action_map: ActionMap) -> Self {
        InputContext {
            name: name.to_string(),
            description: description.to_string(),
            action_map,
        }
    }
}

#[derive(Default, Component)]
#[storage(HashMapStorage)]
pub struct MappedInput {
    pub actions: Vec<ActionId>,
}

impl MappedInput {
    pub fn new() -> Self {
        MappedInput {
            actions: Vec::new(),
        }
    }
}

struct InputManager;

impl<'a> System<'a> for InputManager {
    type SystemData = (
        ReadStorage<'a, InputContext>,
        Read<'a, WindowEvents>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, events, mut mapped): Self::SystemData) {
        let mut event_used: Vec<bool> = events.0.iter().map(|_| false).collect::<Vec<_>>();

        // TODO: Sort on prio first
        for (ctx, mi) in (&contexts, &mut mapped).join() {
            mi.actions = Vec::new();
            for (idx, event) in events.0.iter().enumerate() {
                if event_used[idx] {
                    continue;
                }

                match event {
                    WindowEvent::KeyboardInput { device_id, input } => {
                        log::debug!(
                            "InputManager: Handling {:?} from device {:?}",
                            input,
                            device_id
                        );
                        input.virtual_keycode.map(|key| {
                            if let Some(&action_id) = ctx.action_map.get(&key) {
                                mi.actions.push(action_id);
                                event_used[idx] = true;
                            }
                        });
                    }
                    e => log::trace!("InputManager: Ignoring {:?}", e),
                }
            }
        }
    }
}

pub const INPUT_MANAGER_SYSTEM_ID: &str = "input_manager_sys";

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(InputManager, INPUT_MANAGER_SYSTEM_ID, &[])
}
