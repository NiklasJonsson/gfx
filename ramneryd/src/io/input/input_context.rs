use crate::ecs::prelude::*;
use std::cmp::Ordering;
use std::collections::hash_map::Entry::*;
use std::collections::HashMap;
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975
//
// Basic idea:
// InputContext acts as a filter for inputs (key press/release, mouse click etc...). There are several InputContexts that consume
// input and if one is not interested in some input, it passes it on to the next.

use super::*;
use ramneryd_derive::Visitable;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Visitable)]
pub enum InputPassthrough {
    Passthrough,
    Consume,
}

pub type ActionMap = HashMap<Button, (ActionId, InputPassthrough)>;
pub type StateMap = HashMap<Button, (StateId, InputPassthrough)>;
pub type AxisConvMap = HashMap<DeviceAxis, (RangeId, Sensitivity, InputPassthrough)>;

// Order is important! Declaration order determines sorting order since PartialOrd and Ord are
// auto derived
#[allow(dead_code)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Visitable)]
pub enum InputContextPriority {
    First,
    Ui,
    DontCare,
}

#[derive(Component, Debug, Visitable)]
#[component(storage = "HashMapStorage")]
pub struct InputContext {
    name: String,
    description: String,
    priority: InputContextPriority,
    action_map: ActionMap,
    state_map: StateMap,
    axis_converter: AxisConvMap,
    wants_cursor_pos: (bool, InputPassthrough),
    wants_text: (bool, InputPassthrough),
    consume_all: bool,
}

impl InputContext {
    pub fn builder(name: &str) -> InputContextBuilder {
        InputContextBuilder::start(name)
    }

    pub fn consume_all(&self) -> bool {
        self.consume_all
    }

    pub fn wants_text(&self) -> (bool, InputPassthrough) {
        self.wants_text
    }

    pub fn wants_cursor_pos(&self) -> (bool, InputPassthrough) {
        self.wants_cursor_pos
    }

    pub fn register_axis_delta(
        &self,
        device_axis: DeviceAxis,
        value: AxisValue,
    ) -> Option<(RangeId, RangeValue, InputPassthrough)> {
        self.axis_converter
            .get(&device_axis)
            .map(|(range_id, sensitivity, passthrough)| {
                (*range_id, sensitivity * value, *passthrough)
            })
    }

    pub fn get_action_for(&self, b: impl Into<Button>) -> Option<&(ActionId, InputPassthrough)> {
        self.action_map.get(&b.into())
    }

    pub fn get_state_for(&self, b: impl Into<Button>) -> Option<&(StateId, InputPassthrough)> {
        self.state_map.get(&b.into())
    }
}

impl Eq for InputContext {}
impl PartialEq for InputContext {
    fn eq(&self, other: &Self) -> bool {
        self.name.eq(&other.name)
    }
}

impl Ord for InputContext {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for InputContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct InputContextBuilder {
    name: String,
    description: Option<String>,
    action_map: ActionMap,
    state_map: StateMap,
    axis_converter: AxisConvMap,
    priority: InputContextPriority,
    wants_cursor_pos: (bool, InputPassthrough),
    wants_text: (bool, InputPassthrough),
    consume_all: bool,
}

#[derive(PartialEq, Debug)]
pub enum InputContextError {
    DuplicateActionForKey(ActionId),
    DuplicateStateForKey(StateId),
    DuplicateRangeForKey(RangeId, Sensitivity),
}

// Struct update syntax
impl InputContextBuilder {
    fn start(name: &str) -> Self {
        InputContextBuilder {
            name: name.to_string(),
            description: None,
            action_map: ActionMap::new(),
            state_map: StateMap::new(),
            axis_converter: AxisConvMap::new(),
            priority: InputContextPriority::DontCare,
            wants_cursor_pos: (false, InputPassthrough::Consume),
            wants_text: (false, InputPassthrough::Consume),
            consume_all: false,
        }
    }

    pub fn priority(self, prio: InputContextPriority) -> Self {
        InputContextBuilder {
            priority: prio,
            ..self
        }
    }

    pub fn description(self, desc: &str) -> Self {
        InputContextBuilder {
            description: Some(desc.to_string()),
            ..self
        }
    }

    pub fn wants_cursor_pos(self, wants_cursor_pos: bool, passthrough: InputPassthrough) -> Self {
        Self {
            wants_cursor_pos: (wants_cursor_pos, passthrough),
            ..self
        }
    }

    pub fn wants_text(self, wants_text: bool, passthrough: InputPassthrough) -> Self {
        Self {
            wants_text: (wants_text, passthrough),
            ..self
        }
    }

    pub fn with_action_passthrough(
        mut self,
        button: impl Into<Button>,
        id: impl Into<ActionId>,
        passthrough: InputPassthrough,
    ) -> Result<Self, InputContextError> {
        match self.action_map.entry(button.into()) {
            Vacant(entry) => {
                entry.insert((id.into(), passthrough));
                Ok(self)
            }
            Occupied(entry) => Err(InputContextError::DuplicateActionForKey(entry.get().0)),
        }
    }

    pub fn with_action(
        self,
        button: impl Into<Button>,
        id: impl Into<ActionId>,
    ) -> Result<Self, InputContextError> {
        self.with_action_passthrough(button, id, InputPassthrough::Consume)
    }

    pub fn with_state_passthrough(
        mut self,
        button: impl Into<Button>,
        id: impl Into<StateId>,
        passthrough: InputPassthrough,
    ) -> Result<Self, InputContextError> {
        match self.state_map.entry(button.into()) {
            Vacant(entry) => {
                entry.insert((id.into(), passthrough));
                Ok(self)
            }
            Occupied(entry) => Err(InputContextError::DuplicateStateForKey(entry.get().0)),
        }
    }

    pub fn with_state(
        self,
        button: impl Into<Button>,
        id: impl Into<StateId>,
    ) -> Result<Self, InputContextError> {
        self.with_state_passthrough(button, id, InputPassthrough::Consume)
    }

    pub fn with_range_passthrough(
        mut self,
        device_axis: DeviceAxis,
        range: impl Into<RangeId>,
        sensitivity: Sensitivity,
        passthrough: InputPassthrough,
    ) -> Result<Self, InputContextError> {
        match self.axis_converter.entry(device_axis) {
            Vacant(entry) => {
                entry.insert((range.into(), sensitivity, passthrough));
                Ok(self)
            }
            Occupied(entry) => {
                let (range, sensi, _) = entry.get();
                Err(InputContextError::DuplicateRangeForKey(*range, *sensi))
            }
        }
    }

    pub fn with_range(
        self,
        device_axis: DeviceAxis,
        range: impl Into<RangeId>,
        sensitivity: Sensitivity,
    ) -> Result<Self, InputContextError> {
        self.with_range_passthrough(device_axis, range, sensitivity, InputPassthrough::Consume)
    }

    pub fn build(self) -> InputContext {
        InputContext {
            name: self.name,
            description: self.description.unwrap_or_else(String::default),
            action_map: self.action_map,
            state_map: self.state_map,
            axis_converter: self.axis_converter,
            priority: self.priority,
            wants_cursor_pos: self.wants_cursor_pos,
            consume_all: self.consume_all,
            wants_text: self.wants_text,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_ctx_default_priority() {
        let c = InputContext::builder("Sys").build();
        assert_eq!(c.priority, InputContextPriority::DontCare);
    }
}
