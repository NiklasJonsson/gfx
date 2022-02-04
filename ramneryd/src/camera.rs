use num_derive::FromPrimitive;

use crate::common::Name;
use crate::ecs;
use crate::io::input::{
    DeviceAxis, Input, InputContext, InputContextError, KeyCode, MappedInput, MouseButton, RangeId,
    Sensitivity, StateId,
};
use crate::math::{Extent, Mat4, Obb, Quat, Transform, Vec3};
use crate::time::Time;
use ecs::prelude::*;

use ramneryd_derive::Visitable;

use num_traits::cast::FromPrimitive;

#[derive(Component, Visitable, Clone, Copy)]
pub struct Camera {
    pub fov_y_radians: f32,
    pub aspect_ratio: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y_radians: std::f32::consts::FRAC_PI_4,
            aspect_ratio: 1.0,
            near: 0.05,
            far: 100.0,
        }
    }
}

impl Camera {
    pub fn proj_matrix(&self) -> Mat4 {
        crate::math::perspective_vk(self.fov_y_radians, self.aspect_ratio, self.near, self.far)
    }

    /// Compute a bounding box, in camera space, covering the view-frustum of this camera
    pub fn view_obb(&self) -> Obb {
        debug_assert!(self.fov_y_radians > 0.0);
        debug_assert!(self.far > self.near);

        let half_height = (self.fov_y_radians / 2.0).tan() * self.far;
        let half_width = self.aspect_ratio * half_height;
        let half_depth = (self.far - self.near) / 2.0;

        let center = Vec3::new(0.0, 0.0, -half_depth - self.near);

        Obb::axis_aligned(
            center,
            Extent {
                w: half_width * 2.0,
                h: half_height * 2.0,
                d: half_depth * 2.0,
            },
        )
    }

    pub fn view_direction(tfm: &Transform) -> Vec3 {
        tfm.rotation
            * Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            }
    }
}

#[derive(Debug)]
pub struct CameraOrientation {
    pub up: Vec3,
    pub view_direction: Vec3,
}

// Avoid gimbal-lock by clamping pitch
const MAX_PITCH: f32 = 0.99 * std::f32::consts::FRAC_PI_2;
const MIN_PITCH: f32 = 0.99 * -std::f32::consts::FRAC_PI_2;
#[derive(Debug, Component, Visitable, Clone, Copy)]
pub struct FreeFlyCameraState {
    yaw: f32,
    pitch: f32,
    speed: f32,
}

impl FreeFlyCameraState {
    fn add_pitch(&mut self, v: f32) {
        self.pitch += v;

        if self.pitch > MAX_PITCH {
            self.pitch = MAX_PITCH;
        }

        if self.pitch < MIN_PITCH {
            self.pitch = MIN_PITCH;
        }
    }

    fn add_yaw(&mut self, v: f32) {
        self.yaw += v;
    }

    fn as_quat(&self) -> Quat {
        (Quat::rotation_y(-self.yaw) * Quat::rotation_x(self.pitch)).normalized()
    }
}

const DEFAULT_MOVEMENT_SPEED: f32 = 2.0;

#[derive(Debug, Copy, Clone, FromPrimitive)]
enum CameraMovement {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
    AllowRotation,
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

#[derive(Default)]
pub struct FreeFlyCameraController;

impl FreeFlyCameraController {
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
        .with_state(MouseButton::Right, AllowRotation)?
        .with_range(DeviceAxis::MouseX, CameraRotation::YawDelta, sens)?
        // Switch y sign since the delta is computed from top-left corner
        .with_range(DeviceAxis::MouseY, CameraRotation::PitchDelta, -sens)?
        .build())
}

impl<'a> ecs::System<'a> for FreeFlyCameraController {
    type SystemData = (
        WriteStorage<'a, MappedInput>,
        WriteStorage<'a, Transform>,
        WriteStorage<'a, FreeFlyCameraState>,
        ReadExpect<'a, Time>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut mapped_inputs, mut transforms, mut cam_states, time) = data;

        for (mi, tfm, state) in (&mut mapped_inputs, &mut transforms, &mut cam_states).join() {
            let mut allow_rotation = false;
            for input in mi.iter() {
                match *input {
                    Input::Range(id, val) => {
                        if allow_rotation {
                            log::trace!("Found range, applying");
                            let rot: CameraRotation = id.into();
                            if rot == CameraRotation::YawDelta {
                                state.add_yaw(val as f32)
                            } else {
                                assert_eq!(rot, CameraRotation::PitchDelta);
                                state.add_pitch(val as f32);
                            }

                            tfm.rotation = state.as_quat();
                        }
                    }
                    Input::State(id) => {
                        if let CameraMovement::AllowRotation = id.into() {
                            allow_rotation = true;
                            continue;
                        }

                        let view_direction = Camera::view_direction(tfm);
                        let up = Vec3 {
                            x: 0.0,
                            y: 1.0,
                            z: 0.0,
                        };
                        use CameraMovement::*;
                        let dir = time.delta_sim()
                            * state.speed
                            * match id.into() {
                                Forward => view_direction,
                                Backward => -view_direction,
                                Left => up.cross(view_direction).normalized(),
                                Right => -up.cross(view_direction).normalized(),
                                Up => up,
                                Down => -up,
                                AllowRotation => unreachable!("Handled separately"),
                            };

                        tfm.position += dir;
                    }
                    _ => unreachable!("No actions for FreeFlyCamera!"),
                }
            }
        }
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(FreeFlyCameraController, "free_fly_camera", &[])
}

pub struct DefaultCamera;

impl crate::Module for DefaultCamera {
    fn load(&mut self, world: &mut World) {
        let t = Transform {
            position: Vec3::new(2.0, 2.0, 2.0),
            ..Default::default()
        };

        let input_context = get_input_context().expect("Unable to create input context");

        world
            .create_entity()
            .with(t)
            .with(input_context)
            .with(Camera::default())
            .with(FreeFlyCameraState {
                yaw: 0.0,
                pitch: 0.0,
                speed: DEFAULT_MOVEMENT_SPEED,
            })
            .with(crate::render::light::ShadowViewer)
            .with(crate::render::MainRenderCamera)
            .with(Name::from(NAME))
            .build();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn check_obb_points<P: Into<Vec3> + Copy>(obb: &Obb, points: &[P], should_contain: bool) {
        for &p in points {
            assert_eq!(obb.contains(p.into()), should_contain);
        }
    }

    #[test]
    fn camera_view_obb() {
        let cam = Camera {
            near: 0.1,
            far: 100.0,
            ..Default::default()
        };

        let obb = cam.view_obb();

        let inside_positions = [
            [-1.2, -0.7, -1.4], // view direction
            [-1.2, -0.7, -0.5],
            [-1.2, -0.7, -0.11],
        ];
        check_obb_points(&obb, &inside_positions, true);

        let outside_positions = [
            [-1.2, -0.7, -0.05],
            [-1.2, -0.7, 1.0],
            [-1.2, -0.7, 10.0],
            [-1.2, -0.7, -101.0],
            [0.0, 0.0, 0.0],
        ];
        check_obb_points(&obb, &outside_positions, false);
    }
}
