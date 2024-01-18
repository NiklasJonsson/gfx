use crate::ecs::prelude::*;

use std::{marker::PhantomData, path::Path};

use self::types::Pos;

// TODO: Recipe error
pub trait Recipe {
    fn apply(&mut self, world: &mut World, entity: Entity);
}

mod types {
    use serde::Deserialize;
    use specs::storage::GenericWriteStorage;

    use crate::ecs::prelude::*;

    use crate::math::{Transform, Vec3};

    #[derive(Deserialize)]
    pub struct Pos {
        x: f32,
        y: f32,
        z: f32,
    }

    impl super::Recipe for Pos {
        fn apply(&mut self, world: &mut World, entity: Entity) {
            let mut tfms = world.write_storage::<Transform>();
            let tfm = tfms
                .get_mut_or_default(entity)
                .expect("Entity that is currently loading does not exist");
            tfm.position = Vec3::new(self.x, self.y, self.z);
        }
    }

    #[derive(Deserialize)]
    pub struct Camera;

    impl super::Recipe for Camera {
        fn apply(&mut self, world: &mut World, entity: Entity) {
            let mut cameras = world.write_storage::<crate::camera::Camera>();
            cameras
                .insert(entity, crate::camera::Camera::default())
                .expect("Failed to apply camera recipe");
        }
    }
}

type Deserializer<'de> = ron::Deserializer<'de>;

trait DeserializeRecipe {
    fn try_deserialize(&self, d: &mut Deserializer<'_>) -> Box<dyn Recipe>;
}

impl<T> DeserializeRecipe for T
where
    T: Recipe + for<'de> serde::de::Deserialize<'de> + 'static,
{
    fn try_deserialize(&self, d: &mut Deserializer<'_>) -> Box<dyn Recipe> {
        let v = T::deserialize(d).expect("Fail");
        Box::new(v)
    }
}

struct RecipeInit<D> {
    try_deserialize: fn(D) -> (D, Option<Box<dyn Recipe>>),
}

impl<'de, D> RecipeInit<D>
where
    D: serde::Deserializer<'de>,
{
    fn new<T>() -> Self
    where
        T: Recipe + serde::de::Deserialize<'de> + 'static,
    {
        Self {
            try_deserialize: |deserializer| match T::deserialize(deserializer) {
                Ok(v) => (deserializer, Some(Box::new(v))),
                Err(_) => (deserializer, None),
            },
        }
    }
}

struct RecipeLookup;

impl<'de> serde::de::DeserializeSeed<'de> for RecipeLookup {
    type Value = Box<dyn Recipe>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let cands: Vec<RecipeInit<D>> = vec![
            RecipeInit::new::<types::Pos>(),
            RecipeInit::new::<types::Camera>(),
        ];

        use serde::Deserialize;
        let p = Pos::deserialize(deserializer)?;
        Ok(Box::new(p))
    }
}

struct RecipeEntityVisitor<'w> {
    world: &'w mut World,
}

impl<'w, 'de> serde::de::Visitor<'de> for RecipeEntityVisitor<'w> {
    type Value = Entity;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "A list of recipes")
    }
    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let entity = self.world.create_entity().build();
        while let Some(mut recipe) = seq.next_element_seed(RecipeLookup)? {
            recipe.apply(self.world, entity);
        }
        Ok(entity)
    }
}

struct RecipeEntityLoader<'a> {
    world: &'a mut World,
}

impl<'a, 'de> serde::de::DeserializeSeed<'de> for RecipeEntityLoader<'a> {
    type Value = Entity;
    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let visitor = RecipeEntityVisitor { world: self.world };
        deserializer.deserialize_seq(visitor)
    }
}

struct RecipeLoader<'a> {
    world: &'a mut World,
    entities: Vec<Entity>,
}

impl<'a, 'de> serde::de::Visitor<'de> for RecipeLoader<'a> {
    type Value = Vec<Entity>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "A list of lists of recipes")
    }

    fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        while let Ok(Some(entity)) = seq.next_element_seed(RecipeEntityLoader { world: self.world })
        {
            self.entities.push(entity);
        }
        Ok(self.entities)
    }
}

pub fn from_str(world: &mut World, ron: &str) -> Vec<Entity> {
    use serde::Deserializer;

    let mut de = ron::Deserializer::from_str(ron).expect("Failed to parse ron");
    let visitor = RecipeLoader {
        world,
        entities: vec![],
    };
    de.deserialize_seq(visitor).expect("Failed to deserialize")
}

pub fn load_file<P>(world: &mut World, path: P) -> std::io::Result<()>
where
    P: AsRef<Path>,
{
    let s = std::fs::read_to_string(path)?;
    from_str(world, &s);
    Ok(())
}

#[cfg(test)]
mod test {
    use specs::WorldExt;

    #[test]
    fn test_recipe_pos() {
        let input = "[
            [
                Pos(x: 0.0, y: 0.0, z: 0.0),
            ]
        ]";

        let mut world = crate::ecs::World::new();
        crate::math::register_components(&mut world);

        let entities = super::from_str(&mut world, input);
        let tfms = world.read_storage::<crate::math::Transform>();
        let tfm = tfms.get(*entities.last().unwrap()).unwrap();
        assert_eq!(tfm.position.x, 0.0);
        assert_eq!(tfm.position.y, 0.0);
        assert_eq!(tfm.position.z, 0.0);
    }
}
