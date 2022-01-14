use crate::ecs::prelude::*;
use serde::{Deserialize, Serialize};

pub type Vec3 = vek::Vec3<f32>;
pub type Vec4 = vek::Vec4<f32>;
pub type Mat4 = vek::Mat4<f32>;
pub type Quat = vek::Quaternion<f32>;
pub type Rgb = vek::Rgb<f32>;
pub type Rgba = vek::Rgba<f32>;
pub type FrustrumPlanes = vek::FrustumPlanes<f32>;

use ramneryd_derive::Visitable;

#[derive(Debug, Copy, Component, Clone, PartialEq, Serialize, Deserialize, Visitable)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: f32,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
            scale: 1.0,
        }
    }

    pub fn pos(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: Vec3 { x, y, z },
            ..Self::identity()
        }
    }
}

impl vek::approx::AbsDiffEq for Transform {
    type Epsilon = f32;
    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.position.abs_diff_eq(&other.position, epsilon)
            && self.rotation.abs_diff_eq(&other.rotation, epsilon)
            && self.scale.abs_diff_eq(&other.scale, epsilon)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<Transform> for Mat4 {
    fn from(t: Transform) -> Self {
        Self::translation_3d(t.position)
            * (Self::from(t.rotation.normalized()) * Self::scaling_3d(t.scale))
    }
}

impl From<Transform> for vek::Transform<f32, f32, f32> {
    fn from(o: Transform) -> Self {
        Self {
            position: o.position,
            orientation: o.rotation.normalized(),
            scale: Vec3::from(o.scale),
        }
    }
}

fn verify_compose(lhs: &Transform, rhs: &Transform, out: &Transform) -> bool {
    let m = Mat4::from(*lhs) * Mat4::from(*rhs);
    let n = Mat4::from(*out);
    use vek::approx::AbsDiffEq;
    if m.abs_diff_ne(&n, 0.0001) {
        log::error!("Mismatch when verfying transform compose");
        log::error!("Expected: {}", m);
        log::error!("Actual: {}", n);
        log::error!("lhs: {:?}", lhs);
        log::error!("rhs: {:?}", rhs);
        return false;
    }
    true
}

impl std::ops::Mul for Transform {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let n = Self {
            position: self.position + (self.rotation * (self.scale * rhs.position)),
            rotation: (self.rotation * rhs.rotation).normalized(),
            scale: self.scale * rhs.scale,
        };

        debug_assert!(
            verify_compose(&self, &rhs, &n),
            "Bad transform multiplication"
        );
        n
    }
}

impl std::ops::MulAssign for Transform {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[derive(Debug, Component, Clone, Copy, Visitable)]
pub struct ModelMatrix(pub Mat4);

impl From<Mat4> for ModelMatrix {
    fn from(m: Mat4) -> Self {
        Self(m)
    }
}

impl From<Transform> for ModelMatrix {
    fn from(t: Transform) -> Self {
        Self(Mat4::from(t))
    }
}

impl From<ModelMatrix> for Mat4 {
    fn from(m: ModelMatrix) -> Self {
        m.0
    }
}

impl std::fmt::Display for ModelMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, Component, Visitable)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn combine(&mut self, other: Self) {
        self.min = Vec3::partial_min(self.min, other.min);
        self.max = Vec3::partial_max(self.max, other.max);
    }
}

impl std::ops::Mul<Aabb> for Mat4 {
    type Output = Aabb;
    fn mul(self, rhs: Aabb) -> Self::Output {
        Aabb {
            min: (self * Vec4::from_point(rhs.min)).xyz(),
            max: (self * Vec4::from_point(rhs.max)).xyz(),
        }
    }
}

impl<T> std::ops::Mul<T> for ModelMatrix
where
    Mat4: std::ops::Mul<T>,
{
    type Output = <Mat4 as std::ops::Mul<T>>::Output;
    fn mul(self, t: T) -> Self::Output {
        self.0 * t
    }
}

pub fn perspective_vk(fov_y_radians: f32, aspect_ratio: f32, near: f32, far: f32) -> Mat4 {
    let mut m = Mat4::perspective_rh_zo(fov_y_radians, aspect_ratio, near, far);
    // vulkan has the y-axis
    // inverted (right-handed upside-down).
    m[(1, 1)] *= -1.0;

    m
}

// https://github.com/PacktPublishing/Vulkan-Cookbook/blob/master/Library/Source%20Files/10%20Helper%20Recipes/05%20Preparing%20an%20orthographic%20projection%20matrix.cpp
pub fn orthographic_vk(planes: FrustrumPlanes) -> Mat4 {
    let mut m = Mat4::identity();
    m[(0, 0)] = 2.0 / (planes.right - planes.left);
    m[(1, 1)] = 2.0 / (planes.bottom - planes.top);
    m[(2, 2)] = 1.0 / (planes.near - planes.far);
    m.cols[3] = Vec4 {
        x: -(planes.right + planes.left) / (planes.right - planes.left),
        y: -(planes.bottom + planes.top) / (planes.bottom - planes.top),
        z: planes.near / (planes.near - planes.far),
        w: 1.0,
    };

    m
}

#[cfg(test)]
mod tests {

    use super::{Mat4, Quat, Transform, Vec3};
    use vek::approx::assert_abs_diff_eq;
    const EPS: f32 = 0.00001;

    fn verify_transform_composition(lhs: Mat4, rhs: Mat4, result: Mat4) {
        let m: Mat4 = lhs * rhs;
        assert_abs_diff_eq!(m, result, epsilon = EPS);
    }

    fn verify_composed(lhs: &Transform, rhs: &Transform, result: &Transform) {
        verify_transform_composition(
            Mat4::from(lhs.rotation),
            Mat4::from(rhs.rotation),
            Mat4::from(result.rotation),
        );
        verify_transform_composition(
            Mat4::scaling_3d(Vec3::from(lhs.scale)),
            Mat4::scaling_3d(Vec3::from(rhs.scale)),
            Mat4::scaling_3d(Vec3::from(result.scale)),
        );
        verify_transform_composition(Mat4::from(*lhs), Mat4::from(*rhs), Mat4::from(*result));

        let vek_tfm_lhs = vek::Transform::from(*lhs);
        let vek_tfm_rhs = vek::Transform::from(*rhs);
        let vek_m = Mat4::from(vek_tfm_lhs) * Mat4::from(vek_tfm_rhs);
        let m = Mat4::from(*lhs * *rhs);
        let n = Mat4::from(*lhs) * Mat4::from(*rhs);
        assert_abs_diff_eq!(vek_m, m, epsilon = EPS);
        assert_abs_diff_eq!(vek_m, n, epsilon = EPS);
    }

    #[test]
    fn compose_pos() {
        let lhs = Transform::pos(1.0, 2.0, 3.0);
        let rhs = Transform::pos(5.0, 6.0, 7.0);

        let out = lhs * rhs;
        assert_abs_diff_eq!(out.position, Vec3::new(6.0, 8.0, 10.0));
    }

    #[test]
    fn compose_pos_ident() {
        let lhs = Transform::identity();
        let rhs = Transform::pos(5.0, 6.0, 7.0);

        let out = lhs * rhs;
        assert_abs_diff_eq!(out, rhs);
    }

    #[test]
    fn compose_rot_ident() {
        let lhs = Transform::identity();
        let rhs = Transform {
            rotation: Quat::rotation_3d(
                std::f32::consts::PI / 2.0,
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ),
            ..Default::default()
        };

        let out = lhs * rhs;
        assert_abs_diff_eq!(out, rhs);
    }

    #[test]
    fn compose_rot() {
        let rot_lhs = Quat::rotation_3d(
            std::f32::consts::PI / 2.0,
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 0.0,
            },
        );
        let rot_rhs = Quat::rotation_3d(
            std::f32::consts::PI / 4.0,
            Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
        );
        let lhs = Transform {
            rotation: rot_lhs,
            ..Default::default()
        };
        let rhs = Transform {
            rotation: rot_rhs,
            ..Default::default()
        };

        let out = lhs * rhs;
        assert_abs_diff_eq!(out.rotation, rot_lhs * rot_rhs);

        let m = Mat4::from(rot_lhs * rot_rhs);
        let n = Mat4::from(rot_lhs) * Mat4::from(rot_rhs);
        assert_abs_diff_eq!(m, n, epsilon = EPS);
        assert_abs_diff_eq!(Mat4::from(out), m, epsilon = EPS);
        verify_composed(&lhs, &rhs, &out);
    }

    #[test]
    fn compose_pos_rot_ident() {
        let lhs = Transform::identity();
        let rhs = Transform {
            rotation: Quat::rotation_3d(
                std::f32::consts::PI / 2.0,
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ),
            position: Vec3::new(23.0, 56.0, 2.0),
            ..Default::default()
        };

        let out = lhs * rhs;
        assert_abs_diff_eq!(out, rhs);
        verify_composed(&lhs, &rhs, &out);
    }

    #[test]
    fn compose_pos_rot() {
        let lhs = Transform {
            rotation: Quat::rotation_3d(
                std::f32::consts::PI / 2.0,
                Vec3 {
                    x: 0.31,
                    y: 1.0,
                    z: 0.0,
                },
            ),
            position: Vec3::new(2.0, 10.0, 100.0),
            ..Default::default()
        };
        let rhs = Transform {
            rotation: Quat::rotation_3d(
                std::f32::consts::PI / 5.32,
                Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.32,
                },
            ),
            position: Vec3::new(-10.0, 56.0, 2.0),
            ..Default::default()
        };

        let out = lhs * rhs;
        verify_composed(&lhs, &rhs, &out);
    }

    #[test]
    fn compose_pos_rot2() {
        let lhs = Transform {
            position: Vec3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
            ..Default::default()
        };
        let rhs = Transform {
            rotation: Quat {
                x: -std::f32::consts::FRAC_1_SQRT_2,
                y: 0.0,
                z: 0.0,
                w: std::f32::consts::FRAC_1_SQRT_2,
            }
            .normalized(),
            ..Default::default()
        };

        let out = lhs * rhs;
        verify_composed(&lhs, &rhs, &out);
    }

    #[test]
    fn compose_pos_rot3() {
        let lhs = Transform {
            position: Vec3 {
                x: 0.0,
                y: 2.0,
                z: 0.0,
            },
            rotation: Quat {
                x: 0.2,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            }
            .normalized(),
            scale: 1.0,
        };
        let rhs = Transform {
            position: Vec3 {
                x: 0.0,
                y: -5.0,
                z: 0.0,
            },
            ..Default::default()
        };
        let result = lhs * rhs;

        verify_composed(&lhs, &rhs, &result);
    }
}
