extern crate nalgebra_glm as glm;
extern crate num_derive;
extern crate winit;

use crate::common::*;
use crate::input;
use crate::input::{DeviceAxis, InputContext, MappedInput, RangeId, Sensitivity, StateId};

use crate::GameState;
use crate::DeltaTime;

use glm::Vec3;
use winit::VirtualKeyCode;

use specs::prelude::*;

use num_traits::cast::FromPrimitive;

use std::time::Duration;

#[derive(Debug)]
pub struct CameraOrientation {
    pub up: Vec3,
    pub direction: Vec3,
}

#[derive(Debug, Component)]
#[storage(HashMapStorage)]
pub struct CameraRotationState {
    yaw: f32,
    pitch: f32,
}

const MOVEMENT_SPEED: f32 = 2.0;

#[derive(Debug, Copy, Clone, FromPrimitive)]
enum CameraMovement {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, Copy, Clone, FromPrimitive, PartialEq, Eq)]
enum CameraRotation {
    YawDelta,
    PitchDelta,
}

// TODO: Nullstorage here?
pub struct FreeFlyCameraController;

impl FreeFlyCameraController {
    pub fn get_orientation_from(rotation_state: &CameraRotationState)
        -> CameraOrientation {
        // Disallow roll => y will only be a function of pitch
        let direction: glm::Vec3 = glm::normalize(&glm::vec3(
            rotation_state.yaw.cos() * rotation_state.pitch.cos(),
            rotation_state.pitch.sin(),
            rotation_state.yaw.sin() * rotation_state.pitch.cos(),
        ));

        // This means Q/E will always be up/down in WORLD coordinates
        // Is this ok? Not 100 % of they best way to fix though.
        let up = glm::vec3(0.0, 1.0, 0.0);

        CameraOrientation{ direction, up }
    }
}

impl<'a> System<'a> for FreeFlyCameraController {
    type SystemData = (
        WriteStorage<'a, MappedInput>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, CameraRotationState>,
        Read<'a, GameState>,
        Read<'a, DeltaTime>,
    );

    fn run(&mut self, data: Self::SystemData) {
        log::trace!("FreeFlyCameraController: run");
        let (
            mut mapped_inputs,
            mut positions,
            mut cam_rot_state,
            game_state,
            delta_time,
        ) = data;

        if *game_state == GameState::Paused {
            log::trace!("Game is paused, camera won't be moved");
            return;
        }

        for (mi, pos, rotation_state) in (
            &mut mapped_inputs,
            &mut positions,
            &mut cam_rot_state,
        )
            .join()
        {
            // REFACTOR:
            //  - When states/actions/ranges are an enum, move into one loop

            for (key, val) in &mi.ranges {
                log::trace!("Found range, applying");
                let rot = CameraRotation::from_u32(*key).unwrap();
                if rot == CameraRotation::YawDelta {
                    rotation_state.yaw += *val as f32;
                } else {
                    assert_eq!(rot, CameraRotation::PitchDelta);
                    rotation_state.pitch += *val as f32;
                }
            }

            let CameraOrientation{direction, up} =
                FreeFlyCameraController::get_orientation_from(&rotation_state);

            for id in &mi.states {
                use CameraMovement::*;
                let dir = *delta_time * MOVEMENT_SPEED
                    * match CameraMovement::from_u32(*id).unwrap() {
                        Forward => direction,
                        Backward => -direction,
                        Left => {
                            glm::normalize(&glm::cross::<f32, glm::U3>(&up, &direction))
                        }
                        Right => {
                            -glm::normalize(&glm::cross::<f32, glm::U3>(&up, &direction))
                        }
                        Up => up,
                        Down => -up,
                    };

                *pos += &dir;
            }

            mi.clear();
        }
    }
}

fn get_input_context() -> InputContext {
    let sens = 0.0005 as Sensitivity;
    use crate::camera::CameraMovement::*;
    InputContext::start("CameraInputContext")
        .with_description("Input mapping for Untethered, 3D camera")
        .with_state(VirtualKeyCode::W, Forward as StateId)
        .with_state(VirtualKeyCode::S, Backward as StateId)
        .with_state(VirtualKeyCode::A, Left as StateId)
        .with_state(VirtualKeyCode::D, Right as StateId)
        .with_state(VirtualKeyCode::E, Up as StateId)
        .with_state(VirtualKeyCode::Q, Down as StateId)
        .with_range(
            DeviceAxis::MouseX,
            CameraRotation::YawDelta as RangeId,
            sens,
        )
        .with_range(
            DeviceAxis::MouseY,
            CameraRotation::PitchDelta as RangeId,
            sens,
        )
        .build()
}

pub fn init_camera(world: &mut World) -> Entity {
    let start_pos = glm::vec3(2.0, 2.0, 2.0);
    let rot_state = CameraRotationState {
        yaw: 0.80,
        pitch: 3.78,
    };

    let input_context = get_input_context();

    let mapped_input = MappedInput::new();

    world
        .create_entity()
        .with::<Position>(start_pos.into())
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
