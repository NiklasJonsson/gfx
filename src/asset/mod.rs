use crate::ecs;

pub mod gltf;

pub fn register_systems<'a, 'b>(
    builder: ecs::ExecutorBuilder<'a, 'b>,
) -> ecs::ExecutorBuilder<'a, 'b> {
    gltf::register_systems(builder)
}
