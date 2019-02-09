use crate::{CurrentFrameWindowEvents, PreviousFrameWindowEvents};
use specs::prelude::*;
use std::collections::HashMap;
use winit::{Event, ElementState, VirtualKeyCode, WindowEvent, WindowEvent::KeyboardInput};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975
//
// Basic idea:
// InputContext acts as a filter for inputs (key press/release, mouse click etc...). There are several InputContexts that consume input and if it's not interested in some input, it passes it on.

pub type ActionId = u32;
pub type StateId = u32;
pub type RawAxis = (f64, f64);
pub type ActionMap = HashMap<VirtualKeyCode, ActionId>;
pub type StateMap = HashMap<VirtualKeyCode, StateId>;

/// A context that may consume some input and map it to an ActionId
// TODO: Range
#[derive(Component)]
#[storage(HashMapStorage)]
pub struct InputContext {
    name: String,
    description: String,
    action_map: ActionMap,
    state_map: StateMap,
}

impl InputContext {
    pub fn start(name: &str) -> InputContextBuilder {
        InputContextBuilder::start(name)
    }
}

pub struct InputContextBuilder {
    name: String,
    description: Option<String>,
    action_map: Option<ActionMap>,
    state_map: Option<StateMap>
}

impl InputContextBuilder {
    pub fn start(name: &str) -> Self {
        InputContextBuilder{
            name: name.to_string(),
            description: None,
            action_map: None,
            state_map: None
        }
    }

    pub fn with_description(self, desc: &str) -> Self {
        InputContextBuilder{
            name: self.name,
            description: Some(desc.to_string()),
            action_map: self.action_map,
            state_map: self.state_map
        }
    }

    pub fn with_action(self, key: VirtualKeyCode, id: ActionId) -> Self {
        // TODO: Error on already inserted
        let mut action_map = match self.action_map {
            None => ActionMap::new(),
            Some(map) => map,
        };

        action_map.insert(key, id);

        return InputContextBuilder{
                name: self.name,
                description: self.description,
                action_map: Some(action_map),
                state_map: self.state_map,
        };
    }

    pub fn with_state(self, key: VirtualKeyCode, id: StateId) -> Self {
        // TODO: Error on already inserted
        let mut state_map = match self.state_map {
            None => StateMap::new(),
            Some(map) => map,
        };

        state_map.insert(key, id);

        return InputContextBuilder{
                name: self.name,
                description: self.description,
                action_map: self.action_map,
                state_map: Some(state_map),
        };
    }

    pub fn build(self) -> InputContext {
        // TODO: Errors
        assert!(self.action_map.is_some() || self.state_map.is_some());
        let action_map = match self.action_map {
            None => ActionMap::new(),
            Some(map) => map
        };

        let state_map = match self.state_map {
            None => StateMap::new(),
            Some(map) => map
        };

        InputContext{
            name: self.name,
            description: self.description.unwrap(),
            action_map,
            state_map,
        }
    }
}

#[derive(Default, Component)]
#[storage(HashMapStorage)]
pub struct MappedInput {
    pub actions: Vec<ActionId>,
    pub states: Vec<StateId>,
}

impl MappedInput {
    // TODO: Use some kind of integer set here. so that we don't have to care about duplicates
    pub fn new() -> Self {
        MappedInput {
            actions: Vec::new(),
            states: Vec::new(),
        }
    }

    fn add_action(&mut self, id: ActionId) {
        if !self.actions.iter().any(|&other| other == id) {
            self.actions.push(id);
        }
    }

    fn set_state(&mut self, id: StateId, state: bool) {
        let idx_opt = self.states.iter().position(|&other| other == id);

        if let Some(idx) = idx_opt {
            if !state {
                self.states.remove(idx);
            }
        } else {
            if state {
                self.states.push(id);
            }
        }
    }
}

struct InputManager;

impl<'a> System<'a> for InputManager {
    type SystemData = (
        ReadStorage<'a, InputContext>,
        Write<'a, CurrentFrameWindowEvents>,
        Write<'a, PreviousFrameWindowEvents>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, mut cur_events, mut prev_events, mut mapped): Self::SystemData) {
        // TODO: Accessing inner type of WindowEvents is not very nice
        log::trace!("InputMapper: Running!");
        let mut event_used: Vec<bool> = cur_events.0.iter().map(|_| false).collect::<Vec<_>>();

        // TODO: Sort on prio first
        for (ctx, mi) in (&contexts, &mut mapped).join() {
            mi.actions = Vec::new();
            for (idx, event) in cur_events.0.iter().enumerate() {
                if event_used[idx] {
                    continue;
                }

                match event {
                    KeyboardInput { device_id, input } => {
                        log::debug!(
                            "InputManager: Handling {:?} from device {:?}",
                            input,
                            device_id
                        );

                        let is_pressed = input.state == ElementState::Pressed;
                        let prev_pressed = prev_events.0.iter().any(|prev_event| match prev_event {
                                KeyboardInput {device_id: prev_device_id, input: prev_input } => {
                                    prev_device_id == device_id &&
                                    prev_input == input &&
                                    prev_input.state == ElementState::Pressed
                                },
                                _ => false,
                            });

                        let is_action = is_pressed && !prev_pressed;


                        if is_action {
                            log::trace!("It's an action!");
                            input.virtual_keycode.map(|key| {
                                if let Some(&action_id) = ctx.action_map.get(&key) {
                                    mi.add_action(action_id);
                                    event_used[idx] = true;
                                } else {
                                    log::trace!("But it was not mapped");
                                }
                            });
                        }

                        log::trace!("Setting state!");
                        // This triggers a state change if it is not an action
                        input.virtual_keycode.map(|key| {
                            if let Some(&id) = ctx.state_map.get(&key) {
                                mi.set_state(id, is_pressed);
                                event_used[idx] = true;
                            } else {
                                log::trace!("But it was not mapped");
                            }
                        });
                    }
                    e => log::trace!("InputManager: Ignoring {:?}", e),
                }
            }
        }

        if !cur_events.0.is_empty() {
            (*prev_events).0.clear();
            (*prev_events).0.append(&mut cur_events.0);
        }
    }
}

pub const INPUT_MANAGER_SYSTEM_ID: &str = "input_manager_sys";

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(InputManager, INPUT_MANAGER_SYSTEM_ID, &[])
}
