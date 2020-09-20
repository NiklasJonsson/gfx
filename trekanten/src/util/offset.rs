use super::Extent2D;

use std::ops::Add;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

impl From<ash::vk::Offset2D> for Offset2D {
    fn from(e: ash::vk::Offset2D) -> Self {
        Self { x: e.x, y: e.y }
    }
}

impl From<Offset2D> for ash::vk::Offset2D {
    fn from(e: Offset2D) -> Self {
        Self { x: e.x, y: e.y }
    }
}

impl std::fmt::Display for Offset2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

impl Add<Extent2D> for Offset2D {
    type Output = Self;
    fn add(self, e: Extent2D) -> Self {
        Self {
            x: self.x + e.width as i32,
            y: self.y + e.height as i32,
        }
    }
}
