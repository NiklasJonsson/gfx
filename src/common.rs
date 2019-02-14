use specs::prelude::*;
use std::ops::AddAssign;

// TODO: Auto derive inner type traits

#[derive(Debug, Component)]
#[storage(VecStorage)]
pub struct Position(glm::Vec3);

impl Position {
    pub fn to_vec3(&self) -> glm::Vec3 {
        self.0
    }
}

impl From<glm::Vec3> for Position {
    fn from(src: glm::Vec3) -> Self {
        Position(src)
    }
}

impl AddAssign<&glm::Vec3> for Position {
    fn add_assign(&mut self, other: &glm::Vec3) {
        self.0 += other;
    }
}

