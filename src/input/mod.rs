use crate::CurrentFrameWindowEvents;
use specs::prelude::*;
use std::collections::{HashMap, HashSet};
use winit::{AxisId, DeviceId, ElementState, KeyboardInput, VirtualKeyCode};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975

mod input_context;
pub use crate::input::input_context::InputContext;
pub use crate::input::input_context::InputContextPriority;

// TODO: Use enum here instead
pub type ActionId = u32;
pub type StateId = u32;
pub type RangeId = u32;

pub type RangeValue = f64;
pub type Sensitivity = f64;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DeviceAxis {
    MouseX,
    MouseY,
}

#[derive(Default, Component, Debug)]
#[storage(HashMapStorage)]
pub struct MappedInput {
    pub actions: HashSet<ActionId>,
    pub states: HashSet<StateId>,
    // TODO: HashSet of tuples?
    pub ranges: HashMap<RangeId, RangeValue>,
}

impl MappedInput {
    pub fn new() -> Self {
        MappedInput {
            actions: HashSet::new(),
            states: HashSet::new(),
            ranges: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        *self = MappedInput::new();
    }

    fn add_action(&mut self, id: ActionId) {
        self.actions.insert(id);
    }

    fn set_state(&mut self, id: StateId, val: bool) {
        if val {
            self.states.insert(id);
        } else {
            let existed = self.states.remove(&id);
            if !existed {
                log::warn!("Called state remove for {}, but it did not exist", id);
            }
        }
    }

    fn set_range_delta(&mut self, id: RangeId, value: RangeValue) {
        // If we get the same id again for this mapped input, just sum the deltas
        // This is fairly unlikely, but lets say the used "moves" the mouse twice during a frame.
        // Then this function will be called twice, and the total movement delta should be passed
        // on to whatever system that wants it.
        *self.ranges.entry(id).or_insert(0.0) += value;
    }
}

// TODO: Put this inside InputManager?
// After all this is not a globally avaiable singleton, but rather internal to the input manager.
// This would be initialized during reigster_systems()
type AxisValue = f64;
#[derive(Debug, Default)]
struct InputManagerState {
    pressed_buttons: HashSet<VirtualKeyCode>,
    axis_movement: HashMap<(DeviceId, AxisId), AxisValue>,
}

impl InputManagerState {
    fn new() -> Self {
        InputManagerState {
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

    /// Return delta compared to previous axis value (0 if this is the first time)
    fn register_axis(&mut self, device: DeviceId, axis: AxisId, value: AxisValue) -> AxisValue {
        self.axis_movement
            .insert((device, axis), value)
            .map_or(0.0, |old| value - old)
    }
}

struct InputManager;

// TODO: Use activeCamera here?
// TODO: Clean this up
// Requirements:
//  - A new "pressed" event for a button should generate both an action and set a state
//  - A new "release" should only update input manager state
impl<'a> System<'a> for InputManager {
    type SystemData = (
        ReadStorage<'a, InputContext>,
        Read<'a, CurrentFrameWindowEvents>,
        Write<'a, InputManagerState>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, cur_events, mut state, mut mapped): Self::SystemData) {
        // TODO: Accessing inner type of WindowEvents is not very nice

        let mut event_used: Vec<bool> = cur_events.0.iter().map(|_| false).collect::<Vec<_>>();

        // Currently we are first handling action + axis->range and then states. This can probably
        // be refactored
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

                            let is_new_press = state.register_key(*input);

                            if is_new_press {
                                log::trace!("It's a new key press!");
                                if let Some(key) = input.virtual_keycode {
                                    if let Some(&action_id) = ctx.get_action_for(key) {
                                        mi.add_action(action_id);
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

        let mut state_used: Vec<bool> = state
            .pressed_buttons
            .iter()
            .map(|_| false)
            .collect::<Vec<_>>();

        for (ctx, mi) in joined {
            for (idx, btn) in state.pressed_buttons.iter().enumerate() {
                if state_used[idx] {
                    continue;
                }

                if let Some(&id) = ctx.get_state_for(*btn) {
                    mi.set_state(id, true);
                    state_used[idx] = true;
                }
            }
        }
    }
}

pub const INPUT_MANAGER_SYSTEM_ID: &str = "input_manager_sys";

pub fn add_resources(world: &mut World) {
    world.add_resource(InputManagerState::new());
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(InputManager, INPUT_MANAGER_SYSTEM_ID, &[])
}
