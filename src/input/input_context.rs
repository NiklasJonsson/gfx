use specs::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use winit::VirtualKeyCode;
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975
//
// Basic idea:
// InputContext acts as a filter for inputs (key press/release, mouse click etc...). There are several InputContexts that consume input and if it's not interested in some input, it passes it on.
// TODO: Maybe InputContext and MappedInput can be forced to always exist together?

use crate::input::*;

type ActionMap = HashMap<VirtualKeyCode, ActionId>;
type StateMap = HashMap<VirtualKeyCode, StateId>;
type AxisConvMap = HashMap<DeviceAxis, (RangeId, Sensitivity)>;

// Order is important! Declaration order determines sorting order since PartialOrd and Ord are
// auto derived
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum InputContextPriority {
    First,
    DontCare,
}

#[derive(Component, Debug)]
#[storage(HashMapStorage)]
pub struct InputContext {
    name: String,
    description: String,
    priority: InputContextPriority,
    action_map: ActionMap,
    state_map: StateMap,
    axis_converter: AxisConvMap,
    consume_all: bool,
}

impl InputContext {
    pub fn start(name: &str) -> InputContextBuilder {
        InputContextBuilder::start(name)
    }

    pub fn set_consume_all(&mut self, val: bool) {
        self.consume_all = val;
    }

    pub fn consume_all(&self) -> bool {
        self.consume_all
    }

    pub fn register_axis_delta(
        &self,
        device_axis: DeviceAxis,
        value: AxisValue,
    ) -> Option<(RangeId, RangeValue)> {
        self.axis_converter
            .get(&device_axis)
            .map(|(range_id, sensitivity)| (*range_id, sensitivity * value))
    }

    pub fn get_action_for(&self, key: VirtualKeyCode) -> Option<&ActionId> {
        self.action_map.get(&key)
    }

    pub fn get_state_for(&self, key: VirtualKeyCode) -> Option<&StateId> {
        self.state_map.get(&key)
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
    consume_all: bool,
}

// TODO: Error handling
impl InputContextBuilder {
    fn start(name: &str) -> Self {
        InputContextBuilder {
            name: name.to_string(),
            description: None,
            action_map: ActionMap::new(),
            state_map: StateMap::new(),
            axis_converter: AxisConvMap::new(),
            priority: InputContextPriority::DontCare,
            consume_all: false,
        }
    }

    pub fn with_priority(self, prio: InputContextPriority) -> Self {
        InputContextBuilder {
            priority: prio,
            ..self
        }
    }

    pub fn with_description(self, desc: &str) -> Self {
        InputContextBuilder {
            description: Some(desc.to_string()),
            ..self
        }
    }

    pub fn with_action(mut self, key: VirtualKeyCode, id: ActionId) -> Self {
        // TODO: Error on already inserted
        self.action_map.insert(key, id);

        self
    }

    pub fn with_state(mut self, key: VirtualKeyCode, id: StateId) -> Self {
        // TODO: Error on already inserted
        self.state_map.insert(key, id);

        self
    }

    pub fn with_range(
        mut self,
        device_axis: DeviceAxis,
        range: RangeId,
        sensitivity: Sensitivity,
    ) -> Self {
        self.axis_converter
            .insert(device_axis, (range, sensitivity));

        self
    }

    pub fn build(self) -> InputContext {
        InputContext {
            name: self.name,
            description: self.description.unwrap(),
            action_map: self.action_map,
            state_map: self.state_map,
            axis_converter: self.axis_converter,
            priority: self.priority,
            consume_all: self.consume_all,
        }
    }
}
