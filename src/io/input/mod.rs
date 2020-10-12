//! Module for mapping input events (key presses, mouse movement etc.) to Actions, States and
//! Ranges, which the caller can define the semantics for. Typical usage:
//! 1. Create an InputContext with the InputContextBuilder, using InputContext::builder() and add
//!    mappings and optionally a priority.
//! 2. Create an entity in the specs World with an InputContext.
//! 3. Store this entity in the System struct
//! 4. When the InputMapper system has run, each entity will have it's mapped input available,
//!    provided the event was not consumed by a InputContext with higher priority.
//! 5. When the System::run is executed, fetch the mapped input with the stored entity.
use specs::prelude::*;
use specs::world::EntitiesRes;
use specs::Component;

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use winit::{event::AxisId, event::DeviceId};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975

mod input_context;
pub use input_context::InputContext;
pub use input_context::InputContextError;
pub use input_context::InputContextPriority;

pub use winit::event::VirtualKeyCode as KeyCode;

#[derive(Default, Debug)]
pub struct CurrentFrameExternalInputs(pub Vec<ExternalInput>);

impl CurrentFrameExternalInputs {
    fn iter(&self) -> impl Iterator<Item = &ExternalInput> {
        self.0.iter()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

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
    contents: HashSet<Input>, // TODO: Should this be a vector instead (to not lose duplicates)?
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
        // Let's say the user "moves" the mouse twice during a frame.
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

pub type AxisValue = f64;

#[derive(Debug)]
pub enum ExternalInput {
    KeyPress(KeyCode),
    KeyRelease(KeyCode),
    MouseDelta { x: AxisValue, y: AxisValue },
}

#[derive(Debug)]
struct InputManager {
    pressed_buttons: HashSet<KeyCode>,
    axis_movement: HashMap<(DeviceId, AxisId), AxisValue>,
}

impl InputManager {
    fn new() -> Self {
        Self {
            pressed_buttons: HashSet::new(),
            axis_movement: HashMap::new(),
        }
    }

    fn register_key_press(&mut self, key: KeyCode) -> bool {
        log::debug!("Registering key press: {:?}", key);
        self.pressed_buttons.insert(key)
    }

    fn register_key_release(&mut self, key: KeyCode) {
        log::debug!("Registering key release: {:?}", key);
        let existed = self.pressed_buttons.remove(&key);
        if !existed {
            log::warn!(
                "Button was released, but was not registered as pressed: {:?}",
                key
            );
        }
    }
}

// Requirements:
//  - A new "pressed" event for a button should generate both an action and set a state
//  - A new "release" should only update input manager internal state
//  - Resend state for keys that remain pressed
impl<'a> System<'a> for InputManager {
    type SystemData = (
        ReadStorage<'a, InputContext>,
        Read<'a, CurrentFrameExternalInputs>,
        Read<'a, EntitiesRes>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, inputs, entities, mut mapped): Self::SystemData) {
        log::trace!("InputManager: run");
        let mut action_keys = VecDeque::with_capacity(inputs.len());
        let mut axes = VecDeque::with_capacity(inputs.len());

        for (_ctx, ent) in (&contexts, &entities).join() {
            if let None = mapped.get(ent) {
                mapped.insert(ent, MappedInput::default()).unwrap();
            }
        }

        for input in inputs.iter() {
            match input {
                ExternalInput::KeyPress(key) => {
                    let is_new_press = self.register_key_press(*key);
                    if is_new_press {
                        log::debug!("New key ({:?}) is an action!", key);
                        action_keys.push_back(key);
                    }
                }
                ExternalInput::KeyRelease(key) => self.register_key_release(*key),
                ExternalInput::MouseDelta { x, y } => {
                    log::debug!("Captured mouse delta ({}, {})", x, y);
                    axes.push_back((DeviceAxis::MouseX, x));
                    axes.push_back((DeviceAxis::MouseY, y));
                }
            }
        }

        let mut state_keys = VecDeque::with_capacity(inputs.len());
        for key in self.pressed_buttons.iter() {
            log::debug!("Key ({:?}) is pressed and will generate a state!", key);
            state_keys.push_back(key);
        }

        let mut joined = (&contexts, &mut mapped).join().collect::<Vec<_>>();
        joined.sort_by(|(ctx_a, _), (ctx_b, _)| ctx_a.cmp(ctx_b));

        log::trace!("Mapping actions");
        for key in action_keys {
            for (ctx, mi) in &mut joined {
                if let Some(&action_id) = ctx.get_action_for(*key) {
                    log::debug!("Mapped to action: {:?}", action_id);
                    mi.add_mapped_action(action_id);
                    break;
                }

                if ctx.consume_all() {
                    break;
                }
            }
        }

        log::trace!("Mapping states");
        for key in state_keys {
            for (ctx, mi) in &mut joined {
                if let Some(&id) = ctx.get_state_for(*key) {
                    mi.add_mapped_state(id);
                    break;
                }

                if ctx.consume_all() {
                    break;
                }
            }
        }

        log::trace!("Mapping ranges");
        for (axis, &axis_delta) in axes {
            for (ctx, mi) in &mut joined {
                if let Some((range_id, range_delta)) = ctx.register_axis_delta(axis, axis_delta) {
                    mi.set_range_delta(range_id, range_delta);
                    break;
                }

                if ctx.consume_all() {
                    break;
                }
            }
        }
    }
}

pub const INPUT_MANAGER_SYSTEM_ID: &str = "input_manager_sys";

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(InputManager::new(), INPUT_MANAGER_SYSTEM_ID, &[])
}

pub fn build_ui<'a>(_world: &World, _ui: &imgui::Ui<'a>, _pos: [f32; 2]) -> [f32; 2] {
    [0.0, 0.0]
}
