use crate::ecs::prelude::*;

#[derive(Default, Component)]
#[component(storage = "NullStorage")]
pub struct Mesh;

pub fn box_mesh() -> Mesh {
    todo!()
}
