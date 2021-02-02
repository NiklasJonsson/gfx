use crate::ecs::prelude::*;

pub type Vec3 = vek::Vec3<f32>;
pub type Vec4 = vek::Vec4<f32>;
pub type Mat4 = vek::Mat4<f32>;
pub type Quat = vek::Quaternion<f32>;

#[derive(Debug, Component, Copy, Clone)]
#[component(inspect)]
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

    pub fn pos(position: Vec3) -> Self {
        Self {
            position,
            ..Self::identity()
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<Transform> for Mat4 {
    fn from(t: Transform) -> Self {
        Self::translation_3d(t.position) * (Self::from(t.rotation) * Self::scaling_3d(t.scale))
    }
}

impl std::ops::Mul for Transform {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            position: rhs.position + (rhs.rotation * (rhs.scale * self.position)),
            rotation: rhs.rotation * self.rotation,
            scale: rhs.scale * self.scale,
        }
    }
}

impl std::ops::MulAssign for Transform {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[derive(Debug, Component, Clone, Copy)]
#[component(inspect)]
pub struct ModelMatrix(pub Mat4);

impl From<ModelMatrix> for [[f32; 4]; 4] {
    fn from(m: ModelMatrix) -> Self {
        m.0.into_col_arrays()
    }
}

impl From<Mat4> for ModelMatrix {
    fn from(x: Mat4) -> Self {
        Self(x)
    }
}

impl Into<Mat4> for ModelMatrix {
    fn into(self) -> Mat4 {
        self.0
    }
}

impl ModelMatrix {
    pub fn identity() -> ModelMatrix {
        Self(Mat4::identity())
    }
}

impl std::fmt::Display for ModelMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, Component)]
#[component(inspect)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn combine(&mut self, other: Self) {
        self.min = Vec3::partial_min(self.min, other.min);
        self.max = Vec3::partial_max(self.max, other.max);
    }
}

impl std::ops::Mul<BoundingBox> for Mat4 {
    type Output = BoundingBox;
    fn mul(self, rhs: BoundingBox) -> Self::Output {
        BoundingBox {
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
