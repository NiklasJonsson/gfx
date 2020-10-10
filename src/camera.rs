use specs::Component;

use num_derive::FromPrimitive;

use crate::io::input::{
    DeviceAxis, Input, InputContext, InputContextError, MappedInput, RangeId, Sensitivity, StateId,
};
use crate::math::{Position, Transform};

use crate::time::DeltaTime;

use glm::Vec3;
use nalgebra_glm as glm;
use winit::event::VirtualKeyCode;

use specs::prelude::*;

use crate::App;

use num_traits::cast::FromPrimitive;

#[derive(Debug)]
pub struct CameraOrientation {
    pub up: Vec3,
    pub view_direction: Vec3,
}

// Avoid gimbal-lock by clamping pitch
const MAX_PITCH: f32 = 0.99 * std::f32::consts::FRAC_PI_2;
const MIN_PITCH: f32 = 0.99 * -std::f32::consts::FRAC_PI_2;
// TODO: Inheret clamping for the fields?
// TOOD: Modulus for yaw?
#[derive(Debug, Component)]
#[storage(HashMapStorage)]
pub struct CameraRotationState {
    yaw: f32,
    pitch: f32,
}

impl CameraRotationState {
    fn clamp(&mut self) {
        if self.pitch > MAX_PITCH {
            self.pitch = MAX_PITCH;
        }

        if self.pitch < MIN_PITCH {
            self.pitch = MIN_PITCH;
        }
    }
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

impl From<StateId> for CameraMovement {
    fn from(id: StateId) -> Self {
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

impl From<RangeId> for CameraRotation {
    fn from(id: RangeId) -> Self {
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

        // If the camera is looking at the origin, view_dir is the vector that
        // goes from the camera towards the origin. Right and up is in relation
        // to the view direction.
        // We need a right vector
        let cam_right = glm::normalize(&glm::cross::<f32, glm::U3>(&view_dir, &up));

        // Create a new up vector for orthonormal basis
        let cam_up = glm::normalize(&glm::cross::<f32, glm::U3>(&cam_right, &view_dir));
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
        let translation_inv = glm::translate(&glm::identity(), &-(pos.0));

        let rotation_inv = glm::mat4(
            cam_right.x,
            cam_right.y,
            cam_right.z,
            0.0,
            cam_up.x,
            cam_up.y,
            cam_up.z,
            0.0,
            -view_dir.x,
            -view_dir.y,
            -view_dir.z,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        rotation_inv * translation_inv
    }

    pub fn set_camera_state(w: &mut World, e: Entity, t: &Transform) {
        log::debug!("Set camera state from transform: {:?}", t);
        assert!(App::entity_has_component::<Self>(w, e));
        assert!(App::entity_has_component::<Camera>(w, e));
        let mat: glm::Mat4 = (*t).into();
        let pos: Position = mat.column(3).xyz().into();
        // TODO: Move this to gltf-specific code. Take pos + view dir + up as args.
        // camera is looking in negative z according to the spec (gltf)
        let view_dir: glm::Vec3 = (mat * glm::vec4(0.0, 0.0, -1.0, 0.0)).xyz().normalize();

        log::debug!("pos: {:?}", pos);

        let mut positions = w.write_storage::<Position>();
        positions
            .insert(e, pos)
            .expect("Could not set position for camera!");

        // These are derived with the same formulas used in get_orientation_from() above
        let pitch = view_dir.y.asin();
        log::debug!("pitch: {}", pitch);

        let yaw_x = (view_dir.x / pitch.cos()).acos();
        let yaw_z = (view_dir.z / pitch.cos()).asin();

        // These should be fairy equal
        log::info!("yaw from view_dir.x: {}", yaw_x);
        log::info!("yaw from view_dir.z: {}", yaw_z);

        let yaw = yaw_x;

        let mut rot_states = w.write_storage::<CameraRotationState>();
        rot_states
            .insert(e, CameraRotationState { yaw, pitch })
            .expect("Could not set rotation state for camera!");
    }
}

// Default input mapping for camera
fn get_input_context() -> Result<InputContext, InputContextError> {
    let sens = 0.005 as Sensitivity;
    use CameraMovement::*;
    Ok(InputContext::builder("CameraInputContext")
        .description("Input mapping for untethered, 3D camera")
        .with_state(VirtualKeyCode::W, Forward)?
        .with_state(VirtualKeyCode::S, Backward)?
        .with_state(VirtualKeyCode::A, Left)?
        .with_state(VirtualKeyCode::D, Right)?
        .with_state(VirtualKeyCode::E, Up)?
        .with_state(VirtualKeyCode::Q, Down)?
        // Switch y since the delta is computed from top-left corner
        .with_range(DeviceAxis::MouseX, CameraRotation::YawDelta, sens)?
        .with_range(DeviceAxis::MouseY, CameraRotation::PitchDelta, -sens)?
        .build())
}

impl<'a> System<'a> for FreeFlyCameraController {
    type SystemData = (
        WriteStorage<'a, MappedInput>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, CameraRotationState>,
        Read<'a, DeltaTime>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut mapped_inputs, mut positions, mut cam_rot_state, delta_time) = data;

        for (mi, pos, rotation_state) in
            (&mut mapped_inputs, &mut positions, &mut cam_rot_state).join()
        {
            for input in mi.iter() {
                match input {
                    Input::Range(id, val) => {
                        log::trace!("Found range, applying");
                        let rot: CameraRotation = (*id).into();
                        if rot == CameraRotation::YawDelta {
                            rotation_state.yaw += *val as f32;
                        } else {
                            assert_eq!(rot, CameraRotation::PitchDelta);
                            rotation_state.pitch += *val as f32;
                        }

                        rotation_state.clamp();
                    }
                    Input::State(id) => {
                        let CameraOrientation { view_direction, up } =
                            FreeFlyCameraController::get_orientation_from(&rotation_state);
                        use CameraMovement::*;
                        let dir = *delta_time
                            * MOVEMENT_SPEED
                            * match (*id).into() {
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
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        let start_pos = Position::new(2.0, 2.0, 2.0);
        // TODO: Compute from bounding box
        let rot_state = CameraRotationState {
            yaw: 4.0,
            pitch: -0.4,
        };

        let input_context = get_input_context().expect("Unable to create input context");

        world
            .create_entity()
            .with(start_pos)
            .with(input_context)
            // Camera marker component means for the ActiveCamera resource
            .with(Camera)
            .with(rot_state)
            .build();
    }
}

pub fn register_systems<'a, 'b>(builder: DispatcherBuilder<'a, 'b>) -> DispatcherBuilder<'a, 'b> {
    builder.with(FreeFlyCameraController, "free_fly_camera", &[])
}
