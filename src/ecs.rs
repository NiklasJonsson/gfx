use specs::prelude::*;
use specs::Component;

pub mod prelude {
    pub use specs::{DenseVecStorage, HashMapStorage, NullStorage, VecStorage};
    pub use specs::{Dispatcher, DispatcherBuilder};
    pub use specs::{Entity, World};
    pub use specs::{Read, ReadStorage, Write, WriteStorage};

    pub use specs::{Builder as _, Join as _, SystemData as _, WorldExt as _};
}

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

pub trait System<'a> {
    type SystemData: specs::SystemData<'a>;

    fn run(&mut self, data: Self::SystemData);
    fn setup(&mut self, world: &mut specs::World) {}
}

struct SpecsSystem<'a, S: System<'a>> {
    s: S,
    _lifetime: std::marker::PhantomData<&'a S>,
}

impl<'a, S: System<'a>> specs::System<'a> for SpecsSystem<'a, S> {
    type SystemData = <S as System<'a>>::SystemData;

    fn run(&mut self, data: <S as System<'a>>::SystemData) {
        log::info!("Running {}", std::any::type_name::<Self>());
        self.s.run(data);
    }

    fn setup(&mut self, world: &mut World) {
        <Self as specs::System>::setup(self, world);
        <S as System>::setup(&mut self.s, world);
    }
}

struct Schedule {}
