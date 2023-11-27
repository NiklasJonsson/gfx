use crate::ecs::prelude::*;

use std::path::Path;

pub trait Recipe {
    fn apply(self, world: &mut World, entity: Entity);
}

mod types {
    use serde::Deserialize;
    use specs::storage::GenericWriteStorage;

    use crate::ecs::prelude::*;

    use crate::math::{Transform, Vec3};

    #[derive(Deserialize)]
    struct Pos {
        x: f32,
        y: f32,
        z: f32,
    }

    impl super::Recipe for Pos {
        fn apply(self, world: &mut World, entity: Entity) {
            let mut tfms = world.write_storage::<Transform>();
            let tfm = tfms
                .get_mut_or_default(entity)
                .expect("Entity that is currently loading does not exist");
            tfm.position = Vec3::new(self.x, self.y, self.z);
        }
    }
}

struct RecipeLoader<'a> {
    world: &'a mut World,
}

struct EntityLoader<'a> {
    world: &'a mut World,
    entity: Entity,
}

impl<'a, 'de> serde::de::DeserializeSeed<'de> for EntityLoader<'a> {
    type Value = Box<dyn Recipe>;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

impl<'a, 'de> serde::de::Visitor<'de> for EntityLoader<'a> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a list")
    }
    fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        Ok(())
    }
}

impl<'a, 'de> serde::de::Visitor<'de> for RecipeLoader<'a> {
    type Value = ();
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a list")
    }
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let entity = self.world.create_entity().build();
        let mut entity_loader = EntityLoader {
            world: self.world,
            entity,
        };
        while let Ok(Some(entity_state)) = seq.next_element_seed(entity_loader) {
            entity_loader.entity = entity_loader.world.create_entity().build();
        }
        Ok(())
    }
}

pub fn from_str(world: &mut World, ron: &str) {
    use serde::Deserializer;

    let mut de = ron::Deserializer::from_str(ron).expect("Failed to parse ron");
    let visitor = RecipeLoader { world };
    de.deserialize_seq(visitor).expect("Failed to deserialize");
}

pub fn load_file<P>(world: &mut World, path: P) -> std::io::Result<()>
where
    P: AsRef<Path>,
{
    let s = std::fs::read_to_string(path)?;
    from_str(world, &s);
    Ok(())
}
