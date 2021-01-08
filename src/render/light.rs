use crate::ecs::prelude::*;
use crate::math::Vec3;

#[derive(Component)]
#[component(inspect)]
pub enum Light {
    Point { color: Vec3, range: f32 },
    Directional { color: Vec3 },
    Spot { color: Vec3, angle: f32 },
}
