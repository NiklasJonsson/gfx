use crate::common::math;
use crate::common::IndexData;
use specs::Entity;

pub mod gltf;

pub struct LoadedAsset {
    pub scene_roots: Vec<Entity>,
    pub camera: Option<math::Transform>,
}

fn generate_line_list_from(index_data: &IndexData) -> IndexData {
    let IndexData(indices) = index_data;
    let mut ret = Vec::new();
    assert_eq!(indices.len() % 3, 0);
    for triangle in indices.chunks(3) {
        ret.push(triangle[0]);
        ret.push(triangle[1]);
        ret.push(triangle[1]);
        ret.push(triangle[2]);
        ret.push(triangle[2]);
        ret.push(triangle[0]);
    }

    IndexData(ret)
}
