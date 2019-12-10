extern crate nalgebra_glm as glm;
extern crate num_derive;
extern crate winit;

use crate::common::*;
use crate::input;
use crate::input::{
    DeviceAxis, Input, InputContext, InputContextError, MappedInput, RangeId, Sensitivity, StateId,
};

use crate::DeltaTime;
use crate::GameState;

use glm::Vec3;
use winit::VirtualKeyCode;

use specs::prelude::*;

use crate::App;

use num_traits::cast::FromPrimitive;

#[derive(Debug)]
pub struct CameraOrientation {
    pub up: Vec3,
    pub view_direction: Vec3,
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

impl From<&StateId> for CameraMovement {
    fn from(id: &StateId) -> Self {
        Self::from_u32(id.0).expect("Error in input context setup, can't convert to CameraMovement")
    }
}

impl Into<StateId> for CameraMovement {
    fn into(self) -> StateId {
        StateId(self as u32)
    }
}

#[derive(Debug, Copy, Clone, FromPrimitive, PartialEq, Eq)]
enum CameraRotation {
    YawDelta,
    PitchDelta,
}

impl From<&RangeId> for CameraRotation {
    fn from(id: &RangeId) -> Self {
        Self::from_u32(id.0).expect("Error in input context setup, can't convert to CameraRotation")
    }
}

impl Into<RangeId> for CameraRotation {
    fn into(self) -> RangeId {
        RangeId(self as u32)
    }
}

/// Generic marker component for any camera type
#[derive(Default, Component)]
#[storage(NullStorage)]
pub struct Camera;

impl Camera {
    pub fn set_camera_state(w: &mut World, e: Entity, t: &Transform) {
        assert!(App::entity_has_component::<FreeFlyCameraController>(w, e));
        FreeFlyCameraController::set_camera_state(w, e, t);
    }
}

#[derive(Default, Component)]
#[storage(NullStorage)]
pub struct FreeFlyCameraController;

impl FreeFlyCameraController {
    pub fn get_orientation_from(rotation_state: &CameraRotationState) -> CameraOrientation {
        // Disallow roll => y will only be a function of pitch
        let view_direction: glm::Vec3 = glm::normalize(&glm::vec3(
            rotation_state.yaw.cos() * rotation_state.pitch.cos(),
            rotation_state.pitch.sin(),
            rotation_state.yaw.sin() * rotation_state.pitch.cos(),
        ));

        // This means Q/E will always be up/down in WORLD coordinates
        let up = glm::vec3(0.0, 1.0, 0.0);
        // TODO: Move the code below to its own struct
        /* Re-enable if we want Q/E to align to upwards/downwards from
         * camera view direction. This needs to be clamped when looking
         * straight down/up though.
        let minus_z = -direction;
        let right: Vec3 = glm::cross::<f32, glm::U3>(&up, &minus_z);
        let up: Vec3= glm::cross::<f32, glm::U3>(&minus_z, &right);
        */

        CameraOrientation { view_direction, up }
    }

    pub fn get_view_matrix_from(pos: &Position, rot_state: &CameraRotationState) -> glm::Mat4 {
        let ori = FreeFlyCameraController::get_orientation_from(rot_state);
        let view_dir = ori.view_direction;
        let up = ori.up;

        // Based on https://learnopengl.com/Getting-started/Camera
        // Which is based on Gram-Schmidt process:
        // https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

        // Reverse direction here as the camera is looking in negative z
        let cam_forward = glm::normalize(&-view_dir);

        // We need a right vector
        let cam_right = glm::normalize(&glm::cross::<f32, glm::U3>(&up, &cam_forward));

        // Create a new up vector for orthonormal basis
        let cam_up = glm::normalize(&glm::cross::<f32, glm::U3>(&cam_forward, &cam_right));
        // cam_transform = T * R, view = inverse(cam_transform) = inv(R) * inv(T)

        /* We could also use the following code:
         {
            let cam_transform = glm::mat4(
                cam_right.x, cam_up.x, cam_forward.x, pos.x(),
                cam_right.y, cam_up.y, cam_forward.y, pos.y(),
                cam_right.z, cam_up.z, cam_forward.z, pos.z(),
                0.0, 0.0, 0.0, 1.0
            );
            glm::inverse(&cam_transform)
         }
         But I'm unsure which is faster and I have not benchmarked.
        */

        // This is the code from the opengl tutorial
        let translation_inv = glm::translate(&glm::identity(), &-(pos.to_vec3()));

        let rotation_inv = glm::mat4(
            cam_right.x,
            cam_right.y,
            cam_right.z,
            0.0,
            cam_up.x,
            cam_up.y,
            cam_up.z,
            0.0,
            cam_forward.x,
            cam_forward.y,
            cam_forward.z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        rotation_inv * translation_inv
    }

    pub fn set_camera_state(w: &mut World, e: Entity, t: &Transform) {
        log::trace!("Set camera state");
        assert!(App::entity_has_component::<Self>(w, e));
        assert!(App::entity_has_component::<Camera>(w, e));
        let mat: glm::Mat4 = (*t).into();
        let pos: Position = mat.column(3).xyz().into();
        let view_dir: glm::Vec3 = mat.column(2).xyz();
        let cam_up: glm::Vec3 = mat.column(1).xyz();

        let mut positions = w.write_storage::<Position>();
        positions
            .insert(e, pos)
            .expect("Could not set position for camera!");

        let pitch = view_dir.y.asin();
        let yaw = (view_dir.x / pitch.cos()).acos();

        log::info!("yaw from view_dir.x: {}", yaw);
        log::info!("yaw from view_dir.<: {}", (view_dir.z / pitch.cos().asin()));

        let mut rot_states = w.write_storage::<CameraRotationState>();
        rot_states
            .insert(e, CameraRotationState{yaw, pitch})
            .expect("Could not set rotation state for camera!");
    }
}

// Default input mapping for camera
fn get_input_context() -> Result<InputContext, InputContextError> {
    let sens = 0.0005 as Sensitivity;
    use CameraMovement::*;
    Ok(InputContext::start("CameraInputContext")
        .with_description("Input mapping for untethered, 3D camera")
        .with_state(VirtualKeyCode::W, Forward)?
        .with_state(VirtualKeyCode::S, Backward)?
        .with_state(VirtualKeyCode::A, Left)?
        .with_state(VirtualKeyCode::D, Right)?
        .with_state(VirtualKeyCode::E, Up)?
        .with_state(VirtualKeyCode::Q, Down)?
        .with_range(DeviceAxis::MouseX, CameraRotation::YawDelta, sens)?
        .with_range(DeviceAxis::MouseY, CameraRotation::PitchDelta, sens)?
        .build())
}

impl<'a> System<'a> for FreeFlyCameraController {
    type SystemData = (
        WriteStorage<'a, MappedInput>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, CameraRotationState>,
        Read<'a, GameState>,
        Read<'a, DeltaTime>,
        ReadStorage<'a, Self>,
    );

    fn run(&mut self, data: Self::SystemData) {
        log::trace!("FreeFlyCameraController: run");
        let (
            mut mapped_inputs,
            mut positions,
            mut cam_rot_state,
            game_state,
            delta_time,
            unique_id,
        ) = data;

        if *game_state == GameState::Paused {
            log::trace!("Game is paused, camera won't be moved");
            return;
        }

        for (mi, pos, rotation_state, _id) in (
            &mut mapped_inputs,
            &mut positions,
            &mut cam_rot_state,
            &unique_id,
        )
            .join()
        {
            for input in mi.iter() {
                match input {
                    Input::Range(id, val) => {
                        log::trace!("Found range, applying");
                        let rot: CameraRotation = id.into();
                        if rot == CameraRotation::YawDelta {
                            rotation_state.yaw += *val as f32;
                        } else {
                            assert_eq!(rot, CameraRotation::PitchDelta);
                            rotation_state.pitch += *val as f32;
                        }
                    }
                    Input::State(id) => {
                        let CameraOrientation { view_direction, up } =
                            FreeFlyCameraController::get_orientation_from(&rotation_state);
                        use CameraMovement::*;
                        let dir = *delta_time
                            * MOVEMENT_SPEED
                            * match id.into() {
                                Forward => view_direction,
                                Backward => -view_direction,
                                Left => glm::normalize(&glm::cross::<f32, glm::U3>(
                                    &up,
                                    &view_direction,
                                )),
                                Right => -glm::normalize(&glm::cross::<f32, glm::U3>(
                                    &up,
                                    &view_direction,
                                )),
                                Up => up,
                                Down => -up,
                            };

                        *pos += &dir;
                    }
                    _ => unreachable!("No actions for FreeFlyCamera!"),
                }
            }
            mi.clear();
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        let start_pos = glm::vec3(2.0, 2.0, 2.0);
        let rot_state = CameraRotationState {
            yaw: 0.80,
            pitch: 3.78,
        };

        let input_context = get_input_context().expect("Unable to create input context");

        let mapped_input = MappedInput::new();

        world
            .create_entity()
            .with::<Position>(start_pos.into())
            .with(input_context)
            .with(mapped_input)
            // Camera marker componenent means for the ActiveCamera resource
            .with(Camera)
            // To ensure we get the right mapped input
            .with(Self)
            .with(rot_state)
            .build();
    }
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(
        FreeFlyCameraController,
        "free_fly_camera",
        &[input::INPUT_MANAGER_SYSTEM_ID],
    )
}
