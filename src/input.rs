use std::collections::HashMap;
use winit::{Event, VirtualKeyCode, WindowEvent};
use log::{debug, info};
// Based primarily on https://www.gamedev.net/articles/programming/general-and-gameplay-programming/designing-a-robust-input-handling-system-for-games-r2975
//
// Basic idea:
// InputContext acts as a filter for inputs (key press/release, mouse click etc...). There are several InputContexts that consume input and if it's not interested in some input, it passes it on.

pub type InputContextId = u32;
pub type ActionId = u32;
pub type Action = (InputContextId, ActionId);
pub type ActionMap = HashMap<VirtualKeyCode, ActionId>;

/// A context that may consume some input and map it to an ActionId
// TODO: State, Range
pub struct InputContext {
    name: String,
    description: String,
    actions: ActionMap,
}

impl InputContext {
    pub fn new(name: String, description: String, actions: ActionMap) -> Self {
        InputContext {
            name,
            description,
            actions,
        }
    }

    fn use_input(&self, input: &VirtualKeyCode) -> Option<&ActionId> {
        self.actions.get(input)
    }
}

impl std::fmt::Display for InputContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.name, self.description)
    }
}

pub struct MappedInput {
    pub actions: Vec<Action>,
}

impl MappedInput {
    fn new() -> Self {
        MappedInput {
            actions: Vec::new(),
        }
    }
}

pub struct InputManager {
    contexts: Vec<(InputContextId, InputContext)>,
    callbacks: Vec<Box<FnMut(&MappedInput)>>,
    next_context_id: InputContextId,
    current_mapped_input: MappedInput,
}

impl InputManager {
    fn get_next_context_id(&mut self) -> InputContextId {
        let id = self.next_context_id;
        self.next_context_id += 1;
        debug!("InputManager: Generated new InputContextId {}", id);
        return id;
    }

    pub fn add_callback<CB: FnMut(&MappedInput) + 'static>(&mut self, callback: CB) {
        debug!("InputManager: Added new callback");
        self.callbacks.push(Box::new(callback));
    }

    pub fn register_input_context(&mut self, input_ctx: InputContext) -> InputContextId {
        debug!("InputManager: Registered new InputContext: {}", input_ctx);
        let id = self.get_next_context_id();
        self.contexts.push((id, input_ctx));
        return id;
    }

    pub fn remove_input_context(&mut self, id: InputContextId) {
        assert!(
            self.contexts
                .iter()
                .filter(|(ctx_id, _)| *ctx_id == id)
                .count()
                == 1
        );
        self.contexts.retain(|(ctx_id, _)| *ctx_id != id);
    }

    pub fn dispatch(&mut self) {
        for callback in &mut self.callbacks {
            (*callback)(&self.current_mapped_input);
        }

        self.current_mapped_input = MappedInput::new();
    }

    pub fn handle_button_input(&mut self, input: VirtualKeyCode) {
        for (ctx_id, ctx) in self.contexts.iter() {
            if let Some(&action_id) = ctx.use_input(&input) {
                self.current_mapped_input.actions.push((*ctx_id, action_id));
                return;
            }
        }
    }

    pub fn new() -> InputManager {
        InputManager {
            contexts: Vec::new(),
            callbacks: Vec::new(),
            next_context_id: 0,
            current_mapped_input: MappedInput::new(),
        }
    }
}
