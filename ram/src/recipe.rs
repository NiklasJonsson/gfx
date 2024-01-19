use nom::sequence::Tuple;

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

pub fn from_str(world: &mut World, input: &str) -> Vec<Entity> {
    let (
        nom::character::complete::char('['),
        nom::character::complete::char(']'),
    )
        .parse(input);

    todo!()
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
