use crate::ecs::prelude::*;

use crate::camera::Camera;
use crate::math::{Mat4, Transform, Vec3};

use super::DebugRendererRes;

#[derive(Clone, Copy, Component)]
#[component(stoage = "NullStorage")]
pub struct DrawFrustum;

const VK_NDC_CUBE: [[f32; 3]; 8] = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

const CUBE_INDICES: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 3, 2, 6, 7, 4, 5, 1, 0];

#[derive(Default)]
struct ViewFrustumDrawer {
    reuse: Vec<Vec3>,
}

impl ViewFrustumDrawer {
    const NAME: &'static str = "ViewFrustumDrawer";
}

impl<'a> System<'a> for ViewFrustumDrawer {
    type SystemData = (
        ReadStorage<'a, Transform>,
        ReadStorage<'a, Camera>,
        ReadStorage<'a, DrawFrustum>,
        ReadExpect<'a, DebugRendererRes>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (transforms, cameras, commands, debug_renderer) = data;
        for (tfm, cam, _) in (&transforms, &cameras, &commands).join() {
            let view_proj = cam.proj_matrix() * Mat4::from(*tfm).inverted();
            let inv = view_proj.inverted();
            self.reuse.clear();
            self.reuse.extend(CUBE_INDICES.into_iter().map(|i| {
                let world_with_w = inv * Vec3::from(VK_NDC_CUBE[i]).with_w(1.0);
                world_with_w.xyz() / world_with_w.w
            }));

            let mut debug_renderer = debug_renderer
                .lock()
                .expect("Failed to get mutex for debug_renderer");
            debug_renderer.draw_line_strip(&self.reuse);
        }
    }
}

pub fn register_systems(builder: crate::ecs::ExecutorBuilder) -> crate::ecs::ExecutorBuilder {
    builder.with(ViewFrustumDrawer::default(), ViewFrustumDrawer::NAME, &[])
}
