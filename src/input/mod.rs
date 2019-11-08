//! Module for mapping input events (key presses, mouse movement etc.) to Actions, States and
//! Ranges, which the caller can define the semantics for. Typical usage:
//! 1. Create an InputContext with the InputContextBuilder, using InputContext::start() and add
//!    mappings and optionally a priority.
//! 2. Create a MappedInput component for storing the mapped Input
//! 3. Create an entity in the specs World with *both* a InputContext and a MappedInput component.
//! 4. When the InputMapper system has run, each entity will have it's mapped input available,
//!    provided the event was not consumed by a InputContext with higher priority.
use crate::CurrentFrameWindowEvents;
use specs::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use winit::{AxisId, DeviceId, ElementState, KeyboardInput, VirtualKeyCode};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975

mod input_context;
pub use crate::input::input_context::InputContext;
pub use crate::input::input_context::InputContextError;
pub use crate::input::input_context::InputContextPriority;

#[derive(Debug, Clone, Copy, Hash, PartialEq)]
pub struct ActionId(pub u32);
#[derive(Debug, Clone, Copy, Hash, PartialEq)]
pub struct StateId(pub u32);
#[derive(Debug, Clone, Copy, Hash, PartialEq)]
pub struct RangeId(pub u32);

pub type RangeValue = f64;
pub type Sensitivity = f64;

/// Action: Single input, e.g. open door, cast spell
/// State: Continous input, e.g. run forward
/// Range: Movement along axis, e.g. mouse
#[derive(Debug, Clone, Copy)]
pub enum Input {
    Action(ActionId),
    State(StateId),
    Range(RangeId, RangeValue),
}

// For PartialEq, Eq and Hash we ignore the RangeValue field for Range.
// This is fine since only the RangeId is required to be unique for each
// input context
impl Hash for Input {
    fn hash<H: Hasher>(&self, state: &mut H) {
        use Input::*;
        match self {
            Action(id) => id.hash(state),
            State(id) => id.hash(state),
            Range(id, _) => id.hash(state),
        }
    }
}

impl Eq for Input {}
impl PartialEq for Input {
    fn eq(&self, other: &Self) -> bool {
        use Input::*;
        match (self, other) {
            (Action(id0), Action(id1)) => id0 == id1,
            (State(id0), State(id1)) => id0 == id1,
            (Range(id0, _), Range(id1, _)) => id0 == id1,
            (_, _) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DeviceAxis {
    MouseX,
    MouseY,
}

#[derive(Default, Component, Debug)]
#[storage(HashMapStorage)]
pub struct MappedInput {
    contents: HashSet<Input>,
}

// Public api
impl MappedInput {
    pub fn new() -> Self {
        Self {
            contents: HashSet::new(),
        }
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn contains_action(&self, id: ActionId) -> bool {
        self.contents.contains(&Input::Action(id))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Input> {
        self.contents.iter()
    }
}

// Private api
impl MappedInput {
    fn add_mapped_action(&mut self, id: ActionId) {
        self.contents.insert(Input::Action(id));
    }

    fn add_mapped_state(&mut self, id: StateId) {
        self.contents.insert(Input::State(id));
    }

    fn set_range_delta(&mut self, id: RangeId, value: RangeValue) {
        // If we get the same id again for this mapped input, just sum the deltas
        // This is fairly unlikely, but lets say the used "moves" the mouse twice during a frame.
        // Then this function will be called twice, and the total movement delta should be passed
        // on to whatever system that wants it.
        let input = Input::Range(id, value);
        // PartialEq for Input is implemented to only look at the id, thus, a Range with
        // another value but the same Id will be found
        let cur = self.contents.get(&input);
        let new = cur.map_or(Input::Range(id, value), |old| match old {
            Input::Range(old_id, old_value) => {
                assert_eq!(old_id, &id);
                Input::Range(id, old_value + value)
            }
            _ => unreachable!("Found something other than Range when using RangeId!"),
        });
        self.contents.insert(new);
    }
}

type AxisValue = f64;

#[derive(Debug)]
struct InputManager {
    pressed_buttons: HashSet<VirtualKeyCode>,
    axis_movement: HashMap<(DeviceId, AxisId), AxisValue>,
}

impl InputManager {
    fn new() -> Self {
        Self {
            pressed_buttons: HashSet::new(),
            axis_movement: HashMap::new(),
        }
    }

    fn register_key(&mut self, input: KeyboardInput) -> bool {
        let is_pressed = input.state == ElementState::Pressed;
        let prev_pressed = input
            .virtual_keycode
            .map_or(false, |key| self.pressed_buttons.contains(&key));

        if let Some(key) = input.virtual_keycode {
            if is_pressed {
                self.pressed_buttons.insert(key);
            } else {
                let existed = self.pressed_buttons.remove(&key);
                if !existed {
                    log::warn!("Button was released, but was not registered as pressed.");
                }
            }
        } else {
            log::warn!(
                "Captured button input, but did not have a virtual key: {:?}",
                input
            );
        }

        is_pressed && !prev_pressed
    }
}

// TODO: Use activeCamera here?
// Requirements:
//  - A new "pressed" event for a button should generate both an action and set a state
//  - A new "release" should only update input manager internal state
impl<'a> System<'a> for InputManager {
    type SystemData = (
        ReadStorage<'a, InputContext>,
        Read<'a, CurrentFrameWindowEvents>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, cur_events, mut mapped): Self::SystemData) {
        let mut event_used: Vec<bool> = cur_events.iter().map(|_| false).collect::<Vec<_>>();

        // First loop handles all events. Input contexts are sorted according to their priority.
        // New key presses are mapped to actions, axis change to range. Key presses are recorded
        // and if they are pressed, they are mapped to States in the second loop
        let mut joined = (&contexts, &mut mapped).join().collect::<Vec<_>>();
        joined.sort_by(|(ctx_a, _), (ctx_b, _)| ctx_a.cmp(ctx_b));
        for (ctx, mi) in &mut joined {
            log::debug!("Mapping for inputcontext: {:?}", ctx);

            for (idx, event) in cur_events.0.iter().enumerate() {
                if event_used[idx] {
                    continue;
                }

                use winit::DeviceEvent::MouseMotion;
                use winit::Event::DeviceEvent;
                use winit::Event::WindowEvent;
                use winit::WindowEvent::KeyboardInput;

                if let WindowEvent {
                    event: inner_event, ..
                } = event
                {
                    match inner_event {
                        KeyboardInput { device_id, input } => {
                            log::debug!(
                                "InputManager: Handling {:?} from device {:?}",
                                input,
                                device_id
                            );

                            let is_new_press = self.register_key(*input);

                            if is_new_press {
                                log::trace!("It's a new key press!");
                                if let Some(key) = input.virtual_keycode {
                                    if let Some(&action_id) = ctx.get_action_for(key) {
                                        mi.add_mapped_action(action_id);
                                        event_used[idx] = true;
                                    } else {
                                        log::trace!("But it was not mapped");
                                    }
                                }
                            }
                        }
                        e => log::debug!("InputManager: Ignoring {:?}", e),
                    }
                } else if let DeviceEvent {
                    event: inner_event, ..
                } = event
                {
                    if let MouseMotion { delta: (x, y) } = inner_event {
                        log::debug!("Captured mouse motion ({}, {})", x, y);

                        let axis_deltas = vec![(DeviceAxis::MouseX, x), (DeviceAxis::MouseY, y)];

                        for (axis, &axis_delta) in axis_deltas.into_iter() {
                            if let Some((range_id, range_delta)) =
                                ctx.register_axis_delta(axis, axis_delta)
                            {
                                mi.set_range_delta(range_id, range_delta);
                                event_used[idx] = true;
                            } else {
                                log::debug!("But x was not registered");
                            }
                        }
                    }
                }

                if ctx.consume_all() {
                    event_used[idx] = true;
                }
            }
        }

        // Send out states as well

        // Keep track of consumed button presses/states so that two contexts can't
        // match on the same pressed button.
        let mut button_used: Vec<bool> = self
            .pressed_buttons
            .iter()
            .map(|_| false)
            .collect::<Vec<_>>();

        for (ctx, mi) in joined {
            for (idx, btn) in self.pressed_buttons.iter().enumerate() {
                if button_used[idx] {
                    continue;
                }

                if let Some(&id) = ctx.get_state_for(*btn) {
                    mi.add_mapped_state(id);
                    button_used[idx] = true;
                }

                if ctx.consume_all() {
                    button_used[idx] = true;
                }
            }
        }
    }
}

pub const INPUT_MANAGER_SYSTEM_ID: &str = "input_manager_sys";

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(InputManager::new(), INPUT_MANAGER_SYSTEM_ID, &[])
}
