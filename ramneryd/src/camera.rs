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
    Boost,
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
    const ID: &'static str = "FreeFlyCameraontroller";
}

// Default input mapping for camera
fn get_input_context() -> Result<InputContext, InputContextError> {
    let sens: Sensitivity = 0.005;
    use CameraMovement::*;
    Ok(InputContext::builder(FreeFlyCameraController::ID)
        .description("Input mapping for untethered, 3D camera")
        .with_state(KeyCode::W, Forward)?
        .with_state(KeyCode::S, Backward)?
        .with_state(KeyCode::A, Left)?
        .with_state(KeyCode::D, Right)?
        .with_state(KeyCode::E, Up)?
        .with_state(KeyCode::Q, Down)?
        .with_state(MouseButton::Right, AllowRotation)?
        .with_state(KeyCode::Space, Boost)?
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
            let mut boost = false;
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
                        use CameraMovement as CM;

                        let view_direction = Camera::view_direction(tfm);
                        let up = Vec3 {
                            x: 0.0,
                            y: 1.0,
                            z: 0.0,
                        };
                        let extra = if boost { 4.0 } else { 1.0 };
                        let dir = time.delta_sim()
                            * extra
                            * state.speed
                            * match id.into() {
                                CM::AllowRotation => {
                                    allow_rotation = true;
                                    continue;
                                }
                                CM::Boost => {
                                    boost = true;
                                    continue;
                                }
                                CM::Forward => view_direction,
                                CM::Backward => -view_direction,
                                CM::Left => up.cross(view_direction).normalized(),
                                CM::Right => -up.cross(view_direction).normalized(),
                                CM::Up => up,
                                CM::Down => -up,
                            };

                        tfm.position += dir;
                    }
                    _ => unreachable!("No actions for FreeFlyCamera!"),
                }
            }
        }
    }
}

pub struct FPSCamera;

impl crate::Module for FPSCamera {
    fn load(&mut self, loader: &mut crate::ModuleLoader) {
        loader.add_system(FreeFlyCameraController, FreeFlyCameraController::ID, &[]);
    }
}

pub struct DefaultCamera;

impl crate::Module for DefaultCamera {
    fn load(&mut self, loader: &mut crate::ModuleLoader) {
        loader.world.register::<Camera>();
        loader
            .world
            .register::<crate::render::light::ShadowViewer>();
        loader.world.register::<crate::render::MainRenderCamera>();
        loader.world.register::<Name>();

        let t = Transform {
            position: Vec3::new(2.0, 2.0, 2.0),
            ..Default::default()
        };

        let input_context = get_input_context().expect("Unable to create input context");

        loader
            .world
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
            .with(Name::from("DefaultCamera"))
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
