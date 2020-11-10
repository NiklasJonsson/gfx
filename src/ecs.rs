use specs::prelude::*;
use specs::Component;

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

pub fn assign<C: specs::Component>(w: &mut World, ent: Entity, c: C) {
    w.write_storage::<C>()
        .insert(ent, c)
        .expect("Failed to assign component");
}

pub fn entity_has_component<C>(w: &World, e: Entity) -> bool
where
    C: specs::Component,
{
    w.read_storage::<C>().get(e).is_some()
}

#[derive(Default, Component)]
#[storage(NullStorage)]
pub struct True<T>
where
    T: Default + Send + Sync + 'static,
{
    _ty: std::marker::PhantomData<T>,
}

#[derive(Default, Component)]
#[storage(NullStorage)]
pub struct False<T>
where
    T: Default + Send + Sync + 'static,
{
    _ty: std::marker::PhantomData<T>,
}

pub struct FlagComponent<T> {
    _ty: std::marker::PhantomData<T>,
}

pub trait Flag {
    type True;
    type False;
}

impl<T> Flag for FlagComponent<T>
where
    T: Default + Send + Sync + 'static,
{
    type True = True<T>;
    type False = False<T>;
}
