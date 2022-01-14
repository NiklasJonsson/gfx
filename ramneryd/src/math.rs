use crate::ecs::prelude::*;
use serde::{Deserialize, Serialize};

pub type Vec3 = vek::Vec3<f32>;
pub type Vec4 = vek::Vec4<f32>;
pub type Mat3 = vek::Mat3<f32>;
pub type Mat4 = vek::Mat4<f32>;
pub type Quat = vek::Quaternion<f32>;
pub type Rgb = vek::Rgb<f32>;
pub type Rgba = vek::Rgba<f32>;
pub type FrustrumPlanes = vek::FrustumPlanes<f32>;
pub type Extent = vek::Extent3<f32>;

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

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub point: Vec3,
    pub normal: Vec3,
}

impl Plane {
    /// Return which half-space the point belongs to (positive if true, negative otherwise). Does not handle points on the plane.
    pub fn halfspace(&self, point: Vec3) -> bool {
        self.normal.dot(point - self.point) > 0.0
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

impl From<&[Vec3]> for Aabb {
    fn from(points: &[Vec3]) -> Self {
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for p in points {
            if p.x < min.x {
                min.x = p.x;
            }

            if p.y < min.y {
                min.y = p.y;
            }

            if p.z < min.z {
                min.z = p.z;
            }

            if p.x > max.x {
                max.x = p.x;
            }

            if p.y > max.y {
                max.y = p.y;
            }

            if p.z > max.z {
                max.z = p.z;
            }
        }

        Self { min, max }
    }
}

impl From<Obb> for Aabb {
    fn from(bb: Obb) -> Self {
        Self::from(bb.corners().as_slice())
    }
}

/// Oriented bounding box
///
/// An oriented bounding box is a box that has an arbitrary orientation (in contrast to [`Aabb`]).
/// It consists of a center point and three orthogonal bounding vectors. Each vector starts at the
/// center and ends at one of the faces.
#[derive(Debug, Clone, Copy, Component, Visitable)]
pub struct Obb {
    u_norm: Vec3,
    u_len: f32,
    v_norm: Vec3,
    v_len: f32,
    w_norm: Vec3,
    w_len: f32,
    center: Vec3,
}

/// Normalize the input vector. Returns the length and the normalized vector.
pub fn normalized(v: Vec3) -> (f32, Vec3) {
    let m = v.magnitude();
    (m, v / m)
}

impl Obb {
    /// Create an oriented bounding box (OBB) from a center position and three vectors,
    /// each from the center point to one of the faces. The vectors need to be orthogonal
    /// (otherwise, this is not a box).
    pub fn new(center: Vec3, u: Vec3, v: Vec3, w: Vec3) -> Self {
        let (u_len, u_norm) = normalized(u);
        let (v_len, v_norm) = normalized(v);
        let (w_len, w_norm) = normalized(w);

        Self {
            u_norm,
            u_len,
            v_norm,
            v_len,
            w_norm,
            w_len,
            center,
        }
    }

    /// Create an (axis-aligned) oriented bounding box (OBB) from a center position and three extents.
    ///
    /// This is similar to an [`Aabb`] but is useful when the bounding box needs to be rotated or
    /// transformed into another coordinate space. An Aabb would not bw valid after a rotation.
    /// If no rotations are needed, prefer using [`Aabb`].
    pub fn axis_aligned(center: Vec3, dims: Extent) -> Self {
        Self::new(
            center,
            Vec3::new(dims.w / 2.0, 0.0, 0.0),
            Vec3::new(0.0, dims.h / 2.0, 0.0),
            Vec3::new(0.0, 0.0, dims.d / 2.0),
        )
    }

    /// Returns the center point of the box.
    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// Returns the three vectors that make up the bounds of the box. Each vector starts at the
    /// center and ends at a face. As such, inverting a vector and adding it to the center gives
    /// the opposite face.
    pub fn uvw(&self) -> [Vec3; 3] {
        [
            self.u_norm * self.u_len,
            self.v_norm * self.v_len,
            self.w_norm * self.w_len,
        ]
    }

    /// Returns the eight corners of the box, in no particular order.
    pub fn corners(&self) -> [Vec3; 8] {
        let [u, v, w] = self.uvw();
        [
            self.center + u + v + w,
            self.center + u + v - w,
            self.center + u - v + w,
            self.center + u - v - w,
            self.center - u + v + w,
            self.center - u + v - w,
            self.center - u - v + w,
            self.center - u - v - w,
        ]
    }

    /// Return whether the point is inside the box.
    ///
    /// If the point is incident with any of the faces of the box (lies in the plane of that face),
    /// the return value is unspecified.
    ///
    /// # Examples
    ///
    /// ```
    /// # use ramneryd::math::{Obb, Vec3};
    /// let obb = Obb::new(
    ///     Vec3::new(0.0, 0.0, 0.0),
    ///     Vec3::new(2.0, 0.0, 0.0),
    ///     Vec3::new(0.0, 2.0, 0.0),
    ///     Vec3::new(0.0, 0.0, 2.0),
    /// );
    /// let p0 = Vec3::new(1.0, 1.0, 1.0);
    /// assert!(obb.contains(p0));
    /// ```
    pub fn contains(&self, point: Vec3) -> bool {
        // A point is outside the box if it is in the positive halfspace of any of the faces.

        let plane = |vec_norm: Vec3, vec_len: f32| -> Plane {
            Plane {
                point: self.center + vec_norm * vec_len,
                normal: vec_norm,
            }
        };

        let planes = [
            plane(self.u_norm, self.u_len),
            plane(-self.u_norm, self.u_len),
            plane(self.v_norm, self.v_len),
            plane(-self.v_norm, self.v_len),
            plane(self.w_norm, self.w_len),
            plane(-self.w_norm, self.w_len),
        ];

        for plane in planes {
            if plane.halfspace(point) {
                return false;
            }
        }

        true
    }
}

impl std::ops::Mul<Obb> for Mat4 {
    type Output = Obb;
    fn mul(self, o: Obb) -> Self::Output {
        let center = self * o.center().with_w(1.0);
        let mut vecs = o.uvw();
        for v in &mut vecs {
            *v = Mat3::from(self) * *v;
        }

        Obb::new(center.xyz(), vecs[0], vecs[1], vecs[2])
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
    // vulkan has the y-axis inverted (right-handed upside-down).
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

    #[test]
    fn plane_halfspace() {
        use super::Plane;
        let plane = Plane {
            point: Vec3::new(0.5, -2.0, -1000.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
        };

        let positive = [
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(10.0, -2.0, 0.0),
            Vec3::new(1337.0, 2.0, -10.0),
            Vec3::new(1337.0, -1232.0, -10.0),
        ];
        for p in positive {
            assert!(plane.halfspace(p));
        }
    }

    #[test]
    fn obb_contains() {
        use super::{Extent, Obb};
        let obb = Obb::axis_aligned(Vec3::new(1.0, 2.0, 3.0), Extent::new(6.0, 4.0, 2.0));

        let inside = [
            [1.0, 2.0, 3.0],
            [2.0, 2.9, 3.0],
            [3.9, 2.9, 3.0],
            [-1.9, 2.9, 3.0],
            [-1.0, 0.1, 2.1],
        ];

        for p in inside {
            assert!(obb.contains(Vec3::from(p)));
        }

        let outside = [
            [5.0, 5.0, 5.1],
            [-2.1, 2.9, 3.0],
            [1.0, 4.1, 3.0],
            [1.0, 2.0, 4.1],
        ];

        for p in outside {
            assert!(!obb.contains(Vec3::from(p)));
        }
    }
}
