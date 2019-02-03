extern crate nalgebra_glm as glm;
extern crate num_derive;
extern crate winit;

use crate::common::*;
use crate::input;
use crate::input::{ActionId, ActionMap, InputContext, MappedInput};

use glm::Vec3;
use winit::VirtualKeyCode;

use specs::prelude::*;

use num_traits::cast::FromPrimitive;
use std::cell::RefCell;
use std::rc::Rc;

// TODO: Use specs-derive instead of manual implementation
#[derive(Debug, Component)]
#[storage(HashMapStorage)]
struct CameraOrientation {
    up: Vec3,
    look_at_pos: Vec3,
}

impl CameraOrientation {
    pub fn new(at: Vec3, up: Vec3) -> CameraOrientation {
        CameraOrientation {
            up,
            look_at_pos: at,
        }
    }
}

const MOVEMENT_SPEED: f32 = 0.02;

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
            MoveLeft => glm::vec3(-1.0, 0.0, 0.0),
            MoveRight => glm::vec3(1.0, 0.0, 0.0),
            MoveUp => glm::vec3(0.0, 1.0, 0.0),
            MoveDown => glm::vec3(0.0, -1.0, 0.0),
        }
    }
}

struct FreeFlyCameraController;

impl<'a> System<'a> for FreeFlyCameraController {
    type SystemData = (
        WriteStorage<'a, MappedInput>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, CameraOrientation>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut mapped_inputs, mut positions, mut camera_orientations) = data;
        // TODO: Handle orientation
        for (mi, pos, _ori) in (&mut mapped_inputs, &mut positions, &mut camera_orientations).join()
        {
            for action_id in &mi.actions {
                let mov_vec: glm::Vec3 =
                    MOVEMENT_SPEED * CameraAction::dir_from_action_id(action_id);
                *pos += &mov_vec;
            }

            mi.actions = Vec::new();
        }
    }
}

fn get_input_context() -> InputContext {
    use crate::camera::CameraAction::*;
    // TODO: Can we generate this in a better way? A builder maybe?
    let mut mappings = ActionMap::new();
    mappings.insert(VirtualKeyCode::W, MoveForward as ActionId);
    mappings.insert(VirtualKeyCode::S, MoveBackward as ActionId);
    mappings.insert(VirtualKeyCode::A, MoveLeft as ActionId);
    mappings.insert(VirtualKeyCode::D, MoveRight as ActionId);
    mappings.insert(VirtualKeyCode::E, MoveUp as ActionId);
    mappings.insert(VirtualKeyCode::Q, MoveDown as ActionId);
    let context = InputContext::new(
        "CameraInputContext",
        "Input mapping for Untethered, 3D camera",
        mappings,
    );

    return context;
}

pub fn init_camera(world: &mut World) -> Entity {
    let start_pos = Position::new(2.0, 2.0, 2.0);
    let orientation = CameraOrientation::new(glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0));

    let input_context = get_input_context();

    let mapped_input = MappedInput::new();

    let entity = world
        .create_entity()
        .with(start_pos)
        .with(orientation)
        .with(input_context)
        .with(mapped_input)
        .build();

    return entity;
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        FreeFlyCameraController,
        "free_fly_camera",
        &[input::INPUT_MANAGER_SYSTEM_ID],
    )
}
