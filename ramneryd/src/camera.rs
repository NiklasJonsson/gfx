use num_derive::FromPrimitive;

use crate::common::Name;
use crate::ecs;
use crate::io::input::{
    DeviceAxis, Input, InputContext, InputContextError, KeyCode, MappedInput, MouseButton, RangeId,
    Sensitivity, StateId,
};
use crate::math::{Mat4, Transform, Vec3};
use crate::time::Time;
use ecs::prelude::*;

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
#[component(storage = "HashMapStorage", inspect)]
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

    Move,
}

impl From<StateId> for CameraMovement {
    fn from(id: StateId) -> Self {
        Self::from_u32(id.0).expect("Error in input context setup, can't convert to CameraMovement")
    }
}

impl From<CameraMovement> for StateId {
    fn from(cm: CameraMovement) -> Self {
        Self(cm as u32)
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

impl From<CameraRotation> for RangeId {
    fn from(cm: CameraRotation) -> Self {
        Self(cm as u32)
    }
}

/// Generic marker component for any camera type
#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct Camera;

/*
impl Camera {
    pub fn set_camera_state(w: &mut World, e: Entity, t: &Transform) {
        FreeFlyCameraController::set_camera_state(w, e, t);
    }
}
*/

#[derive(Default)]
pub struct FreeFlyCameraController;

impl FreeFlyCameraController {
    pub fn get_orientation_from(rotation_state: &CameraRotationState) -> CameraOrientation {
        // Disallow roll => y will only be a function of pitch
        let view_direction = Vec3::new(
            rotation_state.yaw.cos() * rotation_state.pitch.cos(),
            rotation_state.pitch.sin(),
            rotation_state.yaw.sin() * rotation_state.pitch.cos(),
        )
        .normalized();

        // This means Q/E will always be up/down in WORLD coordinates
        let up = Vec3::new(0.0, 1.0, 0.0);
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

    pub fn get_view_matrix_from(pos: Vec3, rot_state: &CameraRotationState) -> Mat4 {
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
        let cam_right = view_dir.cross(up).normalized();

        // Create a new up vector for orthonormal basis
        let cam_up = cam_right.cross(view_dir).normalized();
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
        let translation_inv = Mat4::translation_3d(-pos);

        let rotation_inv = Mat4::new(
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

    /*
    pub fn set_camera_state(w: &mut World, e: Entity, t: &Transform) {
        log::debug!("Set camera state from transform: {:?}", t);
        assert!(ecs::entity_has_component::<Camera>(w, e));
        let mat: Mat4 = (*t).into();
        // TODO: Move this to gltf-specific code. Take pos + view dir + up as args.
        // camera is looking in negative z according to the spec (gltf)
        let view_dir: Vec3 = (mat * Vec4::new(0.0, 0.0, -1.0, 0.0)).xyz().normalized();

        let mut transforms = w.write_storage::<Transform>();
        transforms
            .insert(e, *t)
            .expect("Could not set transform for camera!");

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
    */
}

const NAME: &str = "FreeFlyCamera";

// Default input mapping for camera
fn get_input_context() -> Result<InputContext, InputContextError> {
    let sens: Sensitivity = 0.005;
    use CameraMovement::*;
    Ok(InputContext::builder(NAME)
        .description("Input mapping for untethered, 3D camera")
        .with_state(KeyCode::W, Forward)?
        .with_state(KeyCode::S, Backward)?
        .with_state(KeyCode::A, Left)?
        .with_state(KeyCode::D, Right)?
        .with_state(KeyCode::E, Up)?
        .with_state(KeyCode::Q, Down)?
        .with_state(MouseButton::Right, Move)?
        // Switch y since the delta is computed from top-left corner
        .with_range(DeviceAxis::MouseX, CameraRotation::YawDelta, sens)?
        .with_range(DeviceAxis::MouseY, CameraRotation::PitchDelta, -sens)?
        .build())
}

impl<'a> ecs::System<'a> for FreeFlyCameraController {
    type SystemData = (
        WriteStorage<'a, MappedInput>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, CameraRotationState>,
        ReadExpect<'a, Time>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut mapped_inputs, mut transforms, mut cam_rot_state, time) = data;

        for (mi, transform, rotation_state) in
            (&mut mapped_inputs, &mut transforms, &mut cam_rot_state).join()
        {
            let mut moving = false;
            for input in mi.iter() {
                match input {
                    Input::Range(id, val) => {
                        if moving {
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
                    }
                    Input::State(id) => {
                        if let CameraMovement::Move = (*id).into() {
                            moving = true;
                            continue;
                        }

                        let CameraOrientation { view_direction, up } =
                            FreeFlyCameraController::get_orientation_from(rotation_state);
                        use CameraMovement::*;
                        let dir = time.delta_sim()
                            * MOVEMENT_SPEED
                            * match (*id).into() {
                                Forward => view_direction,
                                Backward => -view_direction,
                                Left => up.cross(view_direction).normalized(),
                                Right => -up.cross(view_direction).normalized(),
                                Up => up,
                                Down => -up,
                                Move => unreachable!("Handled separately"),
                            };

                        transform.position += dir;
                    }
                    _ => unreachable!("No actions for FreeFlyCamera!"),
                }
            }
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        let t = Transform {
            position: Vec3::new(2.0, 2.0, 2.0),
            ..Default::default()
        };
        // TODO: Compute from bounding box
        let rot_state = CameraRotationState {
            yaw: 4.0,
            pitch: -0.4,
        };

        let input_context = get_input_context().expect("Unable to create input context");

        world
            .create_entity()
            .with(t)
            .with(input_context)
            .with(Camera)
            .with(rot_state)
            .with(Name::from(NAME))
            .build();
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(FreeFlyCameraController, "free_fly_camera", &[])
}
