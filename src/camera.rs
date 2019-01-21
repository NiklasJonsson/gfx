extern crate nalgebra_glm as glm;
extern crate num_derive;
extern crate winit;

use crate::input::{ActionId, ActionMap, InputContext, InputContextId, InputManager, MappedInput};
use glm::Vec3;
use winit::VirtualKeyCode;

use num_traits::cast::FromPrimitive;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub struct Camera {
    position: Vec3,
    direction: Vec3,
    up: Vec3,
}

const MOVEMENT_SPEED: f32 = 0.02;
impl Camera {
    pub fn new(position: [f32; 3], direction: [f32; 3]) -> Self {
        Camera {
            position: position.into(),
            direction: direction.into(),
            up: glm::vec3(0.0, 1.0, 0.0),
        }
    }

    pub fn move_in_direction(&mut self, dir: Vec3) {
        self.position += dir * MOVEMENT_SPEED;
    }

    pub fn get_pos(&self) -> &Vec3 {
        return &self.position;
    }

    pub fn get_up(&self) -> &Vec3 {
        return &self.up;
    }
}

// TODO: Change Rc to reference?
pub struct CameraController {
    context_id: InputContextId,
    camera: Rc<RefCell<Camera>>,
}

#[derive(Debug, Copy, Clone, FromPrimitive)]
enum CameraAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
}

impl CameraAction {
    // TODO: Can we avoid knowing that ActionId is u32?
    fn dir_from_action_id(id: &ActionId) -> Vec3 {
        use crate::camera::CameraAction::*;
        match CameraAction::from_u32(*id).expect("Invalid action id, not a CameraAction") {
            MoveForward => glm::vec3(0.0, 0.0, 1.0),
            MoveBackward => glm::vec3(0.0, 0.0, -1.0),
            MoveLeft => glm::vec3(1.0, 0.0, 0.0),
            MoveRight => glm::vec3(-1.0, 0.0, 0.0),
            MoveUp => glm::vec3(0.0, 1.0, 0.0),
            MoveDown => glm::vec3(0.0, -1.0, 0.0),
        }
    }
}

impl CameraController {
    fn handle_actions(&self, mapped_input: &MappedInput) {
        for (ctx_id, action_id) in &mapped_input.actions {
            if *ctx_id == self.context_id {
                self.camera
                    .borrow_mut()
                    .move_in_direction(CameraAction::dir_from_action_id(action_id));
            }
        }
    }

    pub fn new(input_manager: &mut InputManager, camera: &Rc<RefCell<Camera>>) -> Rc<Self> {
        use crate::camera::CameraAction::*;
        let mut mappings = ActionMap::new();
        mappings.insert(VirtualKeyCode::W, MoveForward as ActionId);
        mappings.insert(VirtualKeyCode::S, MoveBackward as ActionId);
        mappings.insert(VirtualKeyCode::A, MoveLeft as ActionId);
        mappings.insert(VirtualKeyCode::D, MoveRight as ActionId);
        mappings.insert(VirtualKeyCode::E, MoveUp as ActionId);
        mappings.insert(VirtualKeyCode::Q, MoveDown as ActionId);
        let context = InputContext::new(
            "CameraInputContext".to_string(),
            "Input mapping for Untethered, 3D camera".to_string(),
            mappings,
        );
        let context_id = input_manager.register_input_context(context);
        let controller = Rc::new(CameraController {
            context_id,
            camera: Rc::clone(camera),
        });
        let cb_ctrl = Rc::clone(&controller);
        input_manager.add_callback(move |x| cb_ctrl.handle_actions(x));
        return controller;
    }
}
