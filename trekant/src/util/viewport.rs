use ash::vk;

pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl From<vk::Viewport> for Viewport {
    fn from(v: vk::Viewport) -> Self {
        Self {
            x: v.x,
            y: v.y,
            width: v.width,
            height: v.height,
            min_depth: v.min_depth,
            max_depth: v.max_depth,
        }
    }
}

impl From<Viewport> for vk::Viewport {
    fn from(v: Viewport) -> Self {
        Self {
            x: v.x,
            y: v.y,
            width: v.width,
            height: v.height,
            min_depth: v.min_depth,
            max_depth: v.max_depth,
        }
    }
}
