use crate::ecs;

pub mod gltf;
pub mod rsf;

pub fn register_systems(builder: ecs::ExecutorBuilder) -> ecs::ExecutorBuilder {
    register_module_systems!(builder, self::gltf, rsf)
}
