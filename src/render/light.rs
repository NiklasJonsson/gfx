use crate::ecs::prelude::*;
use crate::math::Vec3;

#[derive(Component)]
#[component(inspect)]
pub enum Light {
    Punctual { color: Vec3 },
    Directional { color: Vec3 },
    Spotlight { color: Vec3, angle: f32 },
}
