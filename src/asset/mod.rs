use crate::math;

pub mod gltf;

pub struct LoadedAsset {
    pub scene_roots: Vec<specs::Entity>,
    pub camera: Option<math::Transform>,
}
