//! Module for mapping input events (key presses, mouse movement etc.) to Actions, States and
//! Ranges, which the caller can define the semantics for. Typical usage:
//! 1. Create an InputContext with the InputContextBuilder, using InputContext::builder() and add
//!    mappings and optionally a priority.
//! 2. Create an entity in the specs World with an InputContext.
//! 3. Store this entity in the System struct
//! 4. When the InputMapper system has run, each entity will have it's mapped input available,
//!    provided the event was not consumed by a InputContext with higher priority.
//! 5. When the System::run is executed, fetch the mapped input with the stored entity.
use crate::ecs::prelude::*;

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use winit::{event::AxisId, event::DeviceId};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975

mod input_context;
pub use input_context::InputContext;
pub use input_context::InputContextError;
pub use input_context::InputContextPriority;
pub use input_context::InputPassthrough;
pub use input_context::{ActionMap, StateMap};

pub use winit::event::MouseButton;
pub use winit::event::VirtualKeyCode as KeyCode;

use ramneryd_derive::Visitable;

#[derive(Default, Debug)]
pub struct CurrentFrameExternalInputs(pub Vec<ExternalInput>);

impl CurrentFrameExternalInputs {
    fn iter(&self) -> impl Iterator<Item = &ExternalInput> {
        self.0.iter()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Visitable)]
pub struct ActionId(pub u32);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Visitable)]
pub struct StateId(pub u32);
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Visitable)]
pub struct RangeId(pub u32);

pub type RangeValue = f64;
pub type Sensitivity = f64;

/// Action: Single input, e.g. open door, cast spell
/// State: Continous input, e.g. run forward
/// Range: Movement along axis, e.g. mouse
#[derive(Debug, Clone)]
pub enum Input {
    Action(ActionId),
    State(StateId),
    Range(RangeId, RangeValue),
    CursorPos(CursorPos),
    Text(Vec<char>),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DeviceAxis {
    MouseX,
    MouseY,
    ScrollX,
    ScrollY,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Button {
    Key(KeyCode),
    Mouse(MouseButton),
}

impl From<KeyCode> for Button {
    fn from(kc: KeyCode) -> Self {
        Self::Key(kc)
    }
}

impl From<MouseButton> for Button {
    fn from(mb: MouseButton) -> Self {
        Self::Mouse(mb)
    }
}

#[derive(Default, Component, Debug)]
#[component(storage = "HashMapStorage")]
pub struct MappedInput {
    contents: Vec<Input>,
}

// Public api
impl MappedInput {
    pub fn iter(&self) -> impl Iterator<Item = &Input> {
        self.contents.iter()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.contents.len()
    }

    pub fn drain(&mut self) -> Vec<Input> {
        std::mem::take(&mut self.contents)
    }
}

// Private api
impl MappedInput {
    fn clear(&mut self) {
        self.contents.clear();
    }

    fn add_action(&mut self, id: ActionId) {
        self.contents.push(Input::Action(id));
    }

    fn add_state(&mut self, id: StateId) {
        self.contents.push(Input::State(id));
    }

    fn add_range_delta(&mut self, id: RangeId, value: RangeValue) {
        self.contents.push(Input::Range(id, value));
    }

    fn add_cursor_pos(&mut self, pos: CursorPos) {
        self.contents.push(Input::CursorPos(pos));
    }

    fn add_text(&mut self, text: Vec<char>) {
        self.contents.push(Input::Text(text));
    }
}

pub type AxisValue = f64;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CursorPos(pub [AxisValue; 2]);

impl CursorPos {
    pub fn x(&self) -> AxisValue {
        self.0[0]
    }

    pub fn y(&self) -> AxisValue {
        self.0[1]
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExternalInput {
    Press(Button),
    Release(Button),
    MouseDelta { x: AxisValue, y: AxisValue },
    ScrollDelta { x: AxisValue, y: AxisValue },
    CursorPos(CursorPos),
    RawChar(char),
}

#[allow(dead_code)]
#[derive(Debug, Default)]
struct InputManager {
    pressed_buttons: HashSet<Button>,
    axis_movement: HashMap<(DeviceId, AxisId), AxisValue>,
}

impl InputManager {
    fn new() -> Self {
        Self::default()
    }

    fn register_key_press(&mut self, button: Button) -> bool {
        log::debug!("Registering press: {:?}", button);
        self.pressed_buttons.insert(button)
    }

    fn register_key_release(&mut self, button: Button) {
        log::debug!("Registering release: {:?}", button);
        let existed = self.pressed_buttons.remove(&button);
        if !existed {
            log::warn!(
                "Button was released, but was not registered as pressed: {:?}",
                button
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
        Write<'a, CurrentFrameExternalInputs>,
        Entities<'a>,
        WriteStorage<'a, MappedInput>,
    );

    fn run(&mut self, (contexts, mut inputs, entities, mut mapped): Self::SystemData) {
        log::trace!("InputManager: run");
        let mut action_keys = Vec::with_capacity(inputs.len());
        let mut axes = Vec::with_capacity(inputs.len());
        let mut chars: Vec<char> = Vec::with_capacity(inputs.len());
        let mut cursor_positions = Vec::with_capacity(inputs.len());

        for (_ctx, ent) in (&contexts, &entities).join() {
            match mapped.entry(ent).unwrap() {
                StorageEntry::Occupied(mut entry) => entry.get_mut().clear(),
                StorageEntry::Vacant(entry) => {
                    entry.insert(MappedInput::default());
                }
            };
        }

        for input in inputs.iter() {
            match input {
                ExternalInput::Press(button) => {
                    let is_new_press = self.register_key_press(*button);
                    if is_new_press {
                        log::debug!("New button ({:?}) is an action!", button);
                        action_keys.push(button);
                    }
                }
                ExternalInput::Release(button) => self.register_key_release(*button),
                ExternalInput::MouseDelta { x, y } => {
                    log::debug!("Captured mouse delta ({}, {})", x, y);
                    axes.push((DeviceAxis::MouseX, x));
                    axes.push((DeviceAxis::MouseY, y));
                }
                ExternalInput::ScrollDelta { x, y } => {
                    axes.push((DeviceAxis::ScrollX, x));
                    axes.push((DeviceAxis::ScrollY, y));
                }
                ExternalInput::RawChar(ch) => chars.push(*ch),
                ExternalInput::CursorPos(pos) => cursor_positions.push(*pos),
            }
        }

        let mut state_keys = Vec::with_capacity(inputs.len());
        for key in self.pressed_buttons.iter() {
            log::debug!("Key ({:?}) is pressed and will generate a state!", key);
            state_keys.push(key);
        }

        let mut joined = (&contexts, &mut mapped).join().collect::<Vec<_>>();
        joined.sort_by(|(ctx_a, _), (ctx_b, _)| ctx_a.cmp(ctx_b));

        log::trace!("Mapping actions");
        for key in action_keys {
            for (ctx, mi) in &mut joined {
                if let Some((action_id, passthrough)) = ctx.get_action_for(*key) {
                    log::debug!("Mapped to action: {:?}", action_id);
                    mi.add_action(*action_id);
                    if let InputPassthrough::Consume = passthrough {
                        break;
                    }
                }

                if ctx.consume_all() {
                    break;
                }
            }
        }

        log::trace!("Mapping states");
        for key in state_keys {
            for (ctx, mi) in &mut joined {
                if let Some((id, passthrough)) = ctx.get_state_for(*key) {
                    mi.add_state(*id);
                    if let InputPassthrough::Consume = passthrough {
                        break;
                    }
                }

                if ctx.consume_all() {
                    break;
                }
            }
        }

        log::trace!("Mapping ranges");
        for (axis, &axis_delta) in axes {
            for (ctx, mi) in &mut joined {
                if let Some((range_id, range_delta, passthrough)) =
                    ctx.register_axis_delta(axis, axis_delta)
                {
                    mi.add_range_delta(range_id, range_delta);
                    if let InputPassthrough::Consume = passthrough {
                        break;
                    }
                }

                if ctx.consume_all() {
                    break;
                }
            }
        }

        log::trace!("Mapping text");
        for (ctx, mi) in &mut joined {
            let (wants_text, passthrough) = ctx.wants_text();
            if wants_text {
                mi.add_text(chars.clone());
                if let InputPassthrough::Consume = passthrough {
                    break;
                }
            }

            if ctx.consume_all() {
                break;
            }
        }

        log::trace!("Mapping cursor positions");
        for (ctx, mi) in &mut joined {
            let (wants_cursor_pos, passthrough) = ctx.wants_cursor_pos();
            if wants_cursor_pos {
                for p in cursor_positions.iter() {
                    mi.add_cursor_pos(*p);
                }
                if let InputPassthrough::Consume = passthrough {
                    break;
                }
            }

            if ctx.consume_all() {
                break;
            }
        }

        inputs.clear();
    }
}

impl InputManager {
    pub const ID: &'static str = "InputManager";
}

pub struct InputModule;

impl crate::Module for InputModule {
    fn load(&mut self, loader: &mut crate::ModuleLoader) {
        loader.world.insert(CurrentFrameExternalInputs(Vec::new()));
        loader.add_system(InputManager::new(), InputManager::ID, &[]);
        // TODO: We shouldn't make a local decision about this.
        loader.add_barrier();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO remaning testing:
    // * Key release/press interaction with action/state
    // * Consume all
    // * Ranges
    // * Ordering of external input vs action/state
    // * Duplicates of mapped input

    #[derive(Debug, Clone, Copy)]
    enum TestState {
        State0,
        State1,
        State2,
    }

    impl From<TestState> for StateId {
        fn from(s: TestState) -> StateId {
            StateId(s as u32)
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum TestAction {
        Action0,
        Action1,
        Action2,
    }

    impl From<TestAction> for ActionId {
        fn from(a: TestAction) -> ActionId {
            ActionId(a as u32)
        }
    }

    /*
    #[derive(Debug, Clone, Copy)]
    enum TestRange {
        DeltaX,
        DeltaY,
    }

    impl From<TestRange> for RangeId {
        fn from(r: TestRange) -> RangeId {
            RangeId(r as u32)
        }
    }
    */

    fn verify_count(world: &World, ent: specs::Entity, i: Input, count: usize) {
        let mapped_inputs = world.write_storage::<MappedInput>();
        let mapped_input = mapped_inputs
            .get(ent)
            .expect("Did not find a mapped input for the ui entity");

        assert!(mapped_input.len() >= count);
        assert_eq!(
            mapped_input
                .iter()
                .filter(|&inp| match (inp, &i) {
                    (Input::Action(aid0), Input::Action(aid1)) => *aid0 == *aid1,
                    (Input::State(sid0), Input::State(sid1)) => *sid0 == *sid1,
                    (Input::Range(rid0, _), Input::Range(rid1, _)) => *rid0 == *rid1,
                    (_, _) => false,
                })
                .count(),
            count
        );
    }

    fn verify_action_count<A: Into<ActionId> + Copy>(
        world: &World,
        ent: specs::Entity,
        a: A,
        count: usize,
    ) {
        verify_count(world, ent, Input::Action(a.into()), count);
    }

    fn verify_state_count<A: Into<StateId> + Copy>(
        world: &World,
        ent: specs::Entity,
        a: A,
        count: usize,
    ) {
        verify_count(world, ent, Input::State(a.into()), count);
    }

    #[test]
    fn input_ctx_sorting() {
        let ctx0 = InputContext::builder("Ctx0")
            .with_action(KeyCode::O, TestAction::Action0)
            .expect("Fail")
            .with_state(KeyCode::X, TestState::State0)
            .expect("Fail")
            .build();

        let ctx1 = InputContext::builder("Ctx1")
            .with_action(KeyCode::O, TestAction::Action1)
            .expect("Fail")
            .with_state(KeyCode::X, TestState::State1)
            .expect("Fail")
            .priority(InputContextPriority::First)
            .build();

        let ctx2 = InputContext::builder("Ctx2")
            .with_action(KeyCode::O, TestAction::Action2)
            .expect("Fail")
            .with_state(KeyCode::X, TestState::State2)
            .expect("Fail")
            .priority(InputContextPriority::Ui)
            .build();

        let contexts = vec![ctx0, ctx1, ctx2];

        let mut executor = ExecutorBuilder::new()
            .with(InputManager::new(), "id", &[])
            .build();
        let mut world = World::new();

        let external_inputs = vec![
            ExternalInput::Press(Button::from(KeyCode::O)),
            ExternalInput::Press(Button::from(KeyCode::X)),
        ];

        executor.setup(&mut world);

        world.insert(CurrentFrameExternalInputs(external_inputs.clone()));

        let entities: Vec<specs::Entity> = contexts
            .into_iter()
            .map(|ctx| world.create_entity().with(ctx).build())
            .collect();

        executor.execute(&world);
        verify_action_count(&world, entities[0], TestAction::Action0, 0);
        verify_action_count(&world, entities[1], TestAction::Action1, 1);
        verify_action_count(&world, entities[2], TestAction::Action2, 0);
        verify_state_count(&world, entities[0], TestState::State0, 0);
        verify_state_count(&world, entities[1], TestState::State1, 1);
        verify_state_count(&world, entities[2], TestState::State2, 0);

        world.insert(CurrentFrameExternalInputs(external_inputs));
        world.delete_entity(entities[1]).expect("Fail");

        executor.execute(&world);
        verify_action_count(&world, entities[0], TestAction::Action0, 0);
        verify_action_count(&world, entities[2], TestAction::Action2, 0);
        verify_state_count(&world, entities[0], TestState::State0, 0);
        verify_state_count(&world, entities[2], TestState::State2, 1);
    }
}
