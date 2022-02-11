use ash::vk;

use super::extent::Extent2D;
use super::offset::Offset2D;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

impl From<vk::Rect2D> for Rect2D {
    fn from(e: vk::Rect2D) -> Self {
        Self {
            offset: Offset2D::from(e.offset),
            extent: Extent2D::from(e.extent),
        }
    }
}

impl From<Rect2D> for vk::Rect2D {
    fn from(e: Rect2D) -> Self {
        Self {
            offset: vk::Offset2D::from(e.offset),
            extent: vk::Extent2D::from(e.extent),
        }
    }
}

impl std::fmt::Display for Rect2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> {} ({})",
            self.offset,
            self.offset + self.extent,
            self.extent
        )
    }
}
