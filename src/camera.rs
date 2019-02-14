extern crate nalgebra_glm as glm;
extern crate num_derive;
extern crate winit;

use crate::common::*;
use crate::input;
use crate::input::{ActionId, InputContext, MappedInput, RangeId, StateId, Sensitivity, DeviceAxis};

use glm::Vec3;
use winit::VirtualKeyCode;

use specs::prelude::*;

use num_traits::cast::FromPrimitive;

#[derive(Debug, Component)]
#[storage(HashMapStorage)]
pub struct CameraOrientation {
    pub up: Vec3,
    pub direction: Vec3,
}

#[derive(Debug, Component)]
#[storage(HashMapStorage)]
struct CameraRotationState {
    yaw: f32,
    pitch: f32,
}

impl CameraOrientation {
    fn new(up: Vec3, direction: Vec3) -> CameraOrientation {
        CameraOrientation {
            up,
            direction,
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

#[derive(Debug, Copy, Clone, FromPrimitive, PartialEq, Eq)]
enum CameraRotation {
    YawDelta,
    PitchDelta,
}

impl CameraAction {
    fn dir_from_state_id(id: StateId) -> Vec3 {
        use crate::camera::CameraAction::*;
        match CameraAction::from_u32(id).expect("Invalid action id, not a CameraAction") {
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
        WriteStorage<'a, CameraRotationState>,
    );

    fn run(&mut self, data: Self::SystemData) {
        log::trace!("FreeFlyCameraController: run");
        let (mut mapped_inputs, mut positions, mut camera_orientations, mut cam_rot_state) = data;
        for (mi, pos, ori, crs) in (&mut mapped_inputs, &mut positions, &mut camera_orientations, &mut cam_rot_state).join()
        {
            // TODO:
            //  - Create a transform class with utlities for this
            //  - Make sure we can move diagonally as well
            //  - Take delta time into account.
            //  - The handling of states/ranges/actions should be in chronological order
            for id in &mi.states {
                use CameraAction::*;
                let dir = MOVEMENT_SPEED * match CameraAction::from_u32(*id).unwrap() {
                    MoveForward => ori.direction,
                    MoveBackward => -ori.direction,
                    MoveLeft => glm::normalize(&glm::cross::<f32, glm::U3>(&ori.up, &ori.direction)),
                    MoveRight => -glm::normalize(&glm::cross::<f32, glm::U3>(&ori.up, &ori.direction)),
                    MoveUp => ori.up,
                    MoveDown => -ori.up,
                };

                *pos += &dir;
            }

            for (key, val) in &mi.ranges {
                log::trace!("Found range, applying");
                let rot = CameraRotation::from_u32(*key).unwrap();
                if rot == CameraRotation::YawDelta {
                    crs.yaw += *val as f32;
                } else {
                    crs.pitch += *val as f32;
                }
            }

            let dir: glm::Vec3 = glm::vec3(crs.yaw.cos() * crs.pitch.cos(),
                                           crs.pitch.sin(),
                                           crs.yaw.sin() * crs.pitch.cos());

            ori.direction = glm::normalize(&dir);

            mi.clear();
        }
    }
}

fn get_input_context() -> InputContext {
    let sens = 0.0005 as Sensitivity;
    use crate::camera::CameraAction::*;
    InputContext::start("CameraInputContext")
        .with_description("Input mapping for Untethered, 3D camera")
        .with_state(VirtualKeyCode::W, MoveForward as ActionId)
        .with_state(VirtualKeyCode::S, MoveBackward as ActionId)
        .with_state(VirtualKeyCode::A, MoveLeft as ActionId)
        .with_state(VirtualKeyCode::D, MoveRight as ActionId)
        .with_state(VirtualKeyCode::E, MoveUp as ActionId)
        .with_state(VirtualKeyCode::Q, MoveDown as ActionId)
        .with_range(DeviceAxis::MouseX, CameraRotation::YawDelta as RangeId, sens)
        .with_range(DeviceAxis::MouseY, CameraRotation::PitchDelta as RangeId, sens)
        .build()
}

pub fn init_camera(world: &mut World) -> Entity {
    let start_pos = glm::vec3(2.0, 2.0, 2.0);
    let direction = glm::normalize(&(glm::vec3(0.0, 0.0, 0.0) - start_pos));
    let up = glm::vec3(0.0, 1.0, 0.0);
    let orientation = CameraOrientation::new(up, direction);
    let rot_state = CameraRotationState{yaw: 0.0, pitch: 0.0};

    let input_context = get_input_context();

    let mapped_input = MappedInput::new();

    world
        .create_entity()
        .with::<Position>(start_pos.into())
        .with(orientation)
        .with(input_context)
        .with(mapped_input)
        .with(rot_state)
        .build()
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        FreeFlyCameraController,
        "free_fly_camera",
        &[input::INPUT_MANAGER_SYSTEM_ID],
    )
}
