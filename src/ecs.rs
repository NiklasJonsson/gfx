use specs::prelude::*;
use specs::Component;

pub type World = specs::World;

pub mod prelude {
    pub use specs::{Component, Entities, Entity};
    pub use specs::{DenseVecStorage, HashMapStorage, NullStorage, VecStorage};
    pub use specs::{Read, ReadStorage, Write, WriteStorage};

    pub use specs::{Builder as _, Join as _, SystemData as _, WorldExt as _};

    pub use super::{Executor, ExecutorBuilder, System, World};
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

pub trait System<'a> {
    type SystemData: specs::SystemData<'a>;

    fn run(&mut self, data: Self::SystemData);
    fn setup(&mut self, _world: &mut specs::World) {}
}

// Too many lifetimes below, not sure how they work. Mostly taken from specs impl.
// It does compile and run though :)

pub struct SpecsSystem<S>
where
    for<'a> S: System<'a> + Sync,
{
    s: S,
}

impl<S> SpecsSystem<S>
where
    for<'a> S: System<'a> + Sync,
{
    pub fn new(s: S) -> Self {
        Self { s }
    }
}

impl<'a, S> specs::System<'a> for SpecsSystem<S>
where
    for<'b> S: System<'b> + Sync,
{
    type SystemData = <S as System<'a>>::SystemData;

    fn run(&mut self, data: <S as System<'a>>::SystemData) {
        log::trace!("Running {}", std::any::type_name::<S>());
        profiling::scope!(std::any::type_name::<S>());
        self.s.run(data);
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        <S as System>::setup(&mut self.s, world);
    }
}

pub struct Executor<'a, 'b> {
    dispatcher: specs::Dispatcher<'a, 'b>,
}

impl<'a, 'b> Executor<'a, 'b> {
    pub fn execute(&mut self, world: &specs::World) {
        self.dispatcher.dispatch(world);
    }

    pub fn setup(&mut self, world: &mut specs::World) {
        self.dispatcher.setup(world);
    }
}

pub struct ExecutorBuilder<'a, 'b> {
    builder: specs::DispatcherBuilder<'a, 'b>,
}

impl<'a, 'b> ExecutorBuilder<'a, 'b> {
    pub fn with<S>(mut self, s: S, id: &str, deps: &[&str]) -> Self
    where
        S: for<'c> System<'c> + Send + 'a + Sync,
    {
        self.builder.add(SpecsSystem::new(s), id, deps);
        self
    }

    pub fn build(self) -> Executor<'a, 'b> {
        Executor {
            dispatcher: self.builder.build(),
        }
    }

    pub fn with_barrier(mut self) -> ExecutorBuilder<'a, 'b> {
        self.builder.add_barrier();
        self
    }

    pub fn new() -> Self {
        Self {
            builder: specs::DispatcherBuilder::new(),
        }
    }
}
