use crate::CurrentFrameWindowEvents;
use specs::prelude::*;
use std::collections::{HashMap, HashSet};
use winit::{AxisId, DeviceId, ElementState, Event, KeyboardInput, VirtualKeyCode};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975
//
// Basic idea:
// InputContext acts as a filter for inputs (key press/release, mouse click etc...). There are several InputContexts that consume input and if it's not interested in some input, it passes it on.

// TODO: Use enum here instead
pub type ActionId = u32;
pub type StateId = u32;
pub type RangeId = u32;
pub type RangeValue = f64;
pub type Sensitivity = f64;

type ActionMap = HashMap<VirtualKeyCode, ActionId>;
type StateMap = HashMap<VirtualKeyCode, StateId>;
type AxisConvMap = HashMap<(DeviceId, AxisId), (RangeId, Sensitivity)>;

/// A context that may consume some input and map it to an ActionId
#[derive(Component)]
#[storage(HashMapStorage)]
pub struct InputContext {
    name: String,
    description: String,
    action_map: ActionMap,
    state_map: StateMap,
    axis_converter: AxisConvMap,
}

impl InputContext {
    pub fn start(name: &str) -> InputContextBuilder {
        InputContextBuilder::start(name)
    }

    fn register_axis_delta(
        &self,
        device: DeviceId,
        axis: AxisId,
        value: AxisValue,
    ) -> Option<(RangeId, RangeValue)> {
        self.axis_converter
            .get(&(device, axis))
            .map(|(range_id, sensitivity)| (*range_id, sensitivity * value))
    }

    fn get_action_for(&self, key: &VirtualKeyCode) -> Option<&ActionId> {
        self.action_map.get(key)
    }

    fn get_state_for(&self, key: &VirtualKeyCode) -> Option<&StateId> {
        self.state_map.get(key)
    }
}

pub struct InputContextBuilder {
    name: String,
    description: Option<String>,
    action_map: Option<ActionMap>,
    state_map: Option<StateMap>,
    axis_converter: Option<AxisConvMap>,
}

impl InputContextBuilder {
    pub fn start(name: &str) -> Self {
        InputContextBuilder {
            name: name.to_string(),
            description: None,
            action_map: None,
            state_map: None,
            axis_converter: None,
        }
    }

    pub fn with_description(self, desc: &str) -> Self {
        InputContextBuilder {
            name: self.name,
            description: Some(desc.to_string()),
            action_map: self.action_map,
            state_map: self.state_map,
            axis_converter: self.axis_converter,
        }
    }

    pub fn with_action(self, key: VirtualKeyCode, id: ActionId) -> Self {
        // TODO: Error on already inserted
        let mut action_map = match self.action_map {
            None => ActionMap::new(),
            Some(map) => map,
        };

        action_map.insert(key, id);

        return InputContextBuilder {
            name: self.name,
            description: self.description,
            action_map: Some(action_map),
            state_map: self.state_map,
            axis_converter: self.axis_converter,
        };
    }

    pub fn with_state(self, key: VirtualKeyCode, id: StateId) -> Self {
        // TODO: Error on already inserted
        let mut state_map = match self.state_map {
            None => StateMap::new(),
            Some(map) => map,
        };

        state_map.insert(key, id);

        return InputContextBuilder {
            name: self.name,
            description: self.description,
            action_map: self.action_map,
            state_map: Some(state_map),
            axis_converter: self.axis_converter,
        };
    }

    pub fn with_range(
        self,
        device: DeviceId,
        axis: AxisId,
        range: RangeId,
        sensitivity: Sensitivity,
    ) -> Self {
        // TODO: Errors
        let mut axis_converter = match self.axis_converter {
            None => AxisConvMap::new(),
            Some(map) => map,
        };

        axis_converter.insert((device, axis), (range, sensitivity));
        return InputContextBuilder {
            name: self.name,
            description: self.description,
            action_map: self.action_map,
            state_map: self.state_map,
            axis_converter: Some(axis_converter),
        };
    }

    pub fn build(self) -> InputContext {
        // TODO: Errors
        // REFACTOR: Can we fix the None/Some pattern somehow? More functional?
        assert!(self.action_map.is_some() || self.state_map.is_some());
        let action_map = match self.action_map {
            None => ActionMap::new(),
            Some(map) => map,
        };

        let state_map = match self.state_map {
            None => StateMap::new(),
            Some(map) => map,
        };

        let axis_converter = match self.axis_converter {
            None => AxisConvMap::new(),
            Some(map) => map,
        };

        InputContext {
            name: self.name,
            description: self.description.unwrap(),
            action_map,
            state_map,
            axis_converter,
        }
    }
}

#[derive(Default, Component)]
#[storage(HashMapStorage)]
pub struct MappedInput {
    pub actions: Vec<ActionId>,
    pub states: Vec<StateId>,
    pub ranges: HashMap<RangeId, RangeValue>,
}

impl MappedInput {
    // TODO: Use some kind of integer set here. so that we don't have to care about duplicates
    pub fn new() -> Self {
        MappedInput {
            actions: Vec::new(),
            states: Vec::new(),
            ranges: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        *self = MappedInput::new();
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

    fn set_range_delta(&mut self, id: RangeId, value: RangeValue) {
        // If we get the same id again for this mapped input, just sum the deltas
        // This is fairly unlikely, but lets say the used "moves" the mouse twice during a frame.
        // Then this function will be called twice, and the total movement delta should be passed
        // on to whatever system that wants it.
        *self.ranges.entry(id).or_insert(0.0) += value;
    }
}

type AxisValue = f64;
type IsPressed = bool;
type PreviousPressed = bool;
#[derive(Debug, Default)]
struct InputManagerState {
    pressed_buttons: HashSet<(DeviceId, VirtualKeyCode)>,
    axis_movement: HashMap<(DeviceId, AxisId), AxisValue>,
}

impl InputManagerState {
    fn new() -> Self {
        InputManagerState {
            pressed_buttons: HashSet::new(),
            axis_movement: HashMap::new(),
        }
    }

    fn register_key(
        &mut self,
        device: DeviceId,
        input: KeyboardInput,
    ) -> (IsPressed, PreviousPressed) {
        let is_pressed = input.state == ElementState::Pressed;
        let prev_pressed = input
            .virtual_keycode
            .map_or(false, |key| self.pressed_buttons.contains(&(device, key)));

        if let Some(key) = input.virtual_keycode {
            let map_key = (device, key);
            if is_pressed {
                self.pressed_buttons.insert(map_key);
            } else {
                let existed = self.pressed_buttons.remove(&map_key);
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

        return (is_pressed, prev_pressed);
    }

    /// Return delta compared to previous axis value (0 if this is the first time)
    fn register_axis(&mut self, device: DeviceId, axis: AxisId, value: AxisValue) -> AxisValue {
        let possible_old_val = self.axis_movement.insert((device, axis), value);
        return possible_old_val.map_or(0.0, |old| value - old);
    }
}

struct InputManager;

impl<'a> System<'a> for InputManager {
    type SystemData = (
        ReadStorage<'a, InputContext>,
        Write<'a, CurrentFrameWindowEvents>,
        Write<'a, InputManagerState>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, mut cur_events, mut state, mut mapped): Self::SystemData) {
        // TODO: Accessing inner type of WindowEvents is not very nice
        log::trace!("InputMapper: Running!");
        let mut event_used: Vec<bool> = cur_events.0.iter().map(|_| false).collect::<Vec<_>>();

        // TODO: Sort on prio first
        for (ctx, mi) in (&contexts, &mut mapped).join() {
            // TODO: Should we clear here?
            mi.clear();

            for (idx, event) in cur_events.0.iter().enumerate() {
                if event_used[idx] {
                    continue;
                }

                use winit::WindowEvent::{AxisMotion, CursorMoved, KeyboardInput};

                match event {
                    KeyboardInput { device_id, input } => {
                        log::debug!(
                            "InputManager: Handling {:?} from device {:?}",
                            input,
                            device_id
                        );

                        let (is_pressed, prev_pressed) = state.register_key(*device_id, *input);
                        let is_action = is_pressed && !prev_pressed;

                        // Both a state and an action is triggered when a button is pressed for the
                        // first time. It is up to the InputContext to not map the same key to an
                        // action and a state at the same time.

                        if is_action {
                            log::trace!("It's an action!");
                            input.virtual_keycode.map(|key| {
                                if let Some(&action_id) = ctx.get_action_for(&key) {
                                    mi.add_action(action_id);
                                    event_used[idx] = true;
                                } else {
                                    log::trace!("But it was not mapped");
                                }
                            });
                        }

                        log::trace!("Setting state!");
                        input.virtual_keycode.map(|key| {
                            if let Some(&id) = ctx.get_state_for(&key) {
                                mi.set_state(id, is_pressed);
                                event_used[idx] = true;
                            } else {
                                log::trace!("But it was not mapped");
                            }
                        });
                    }
                    AxisMotion {
                        device_id,
                        axis,
                        value,
                    } => {
                        log::debug!("Captured axis motion {:?}", event);
                        let delta = state.register_axis(*device_id, *axis, *value);
                        if let Some((id, val)) = ctx.register_axis_delta(*device_id, *axis, delta) {
                            mi.set_range_delta(id, val);
                            event_used[idx] = true;
                        } else {
                            log::trace!("But it was not registered");
                        }
                    }
                    e => log::trace!("InputManager: Ignoring {:?}", e),
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
