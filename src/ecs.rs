use specs::prelude::*;

pub fn get_singleton_entity<C>(w: &World) -> Entity
where
    C: specs::Component,
{
    let markers = w.read_storage::<C>();
    let entities = w.read_resource::<specs::world::EntitiesRes>();

    let mut joined = (&entities, &markers).join();
    let item = joined.next();
    assert!(
        joined.next().is_none(),
        "Expected only one entity with marker component!"
    );
    let (ent, _) = item.expect("Expected an entity!");

    ent
}
