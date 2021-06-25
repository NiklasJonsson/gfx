use crate::ecs;

pub mod gltf;
pub mod rsf;

pub fn register_systems<'a, 'b>(
    builder: ecs::ExecutorBuilder<'a, 'b>,
) -> ecs::ExecutorBuilder<'a, 'b> {
    register_module_systems!(builder, self::gltf, rsf)
}
