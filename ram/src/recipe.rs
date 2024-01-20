use serde::{Deserialize, Serialize};

use crate::ecs::prelude::*;
use crate::math::{Transform, Vec3};

use std::path::Path;

#[derive(Debug)]
pub enum RecipeError {
    DeadEntity(Entity),
    InvalidEntity(Entity, specs::error::Error),
}

impl std::fmt::Display for RecipeError {
    fn fmt<'a>(&self, fmt: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        // TODO
        write!(fmt, "{:?}", self)
    }
}

pub type Result = std::result::Result<(), RecipeError>;

#[typetag::serde]
pub trait Recipe {
    fn apply(&mut self, world: &mut World, entity: Entity) -> Result;
}

fn dead_entity_err(entity: Entity) -> RecipeError {
    RecipeError::DeadEntity(entity)
}

fn specs_error(entity: Entity, error: specs::error::Error) -> RecipeError {
    RecipeError::InvalidEntity(entity, error)
}
#[derive(Deserialize, Serialize)]
pub struct Pos {
    x: f32,
    y: f32,
    z: f32,
}

#[typetag::serde]
impl Recipe for Pos {
    fn apply(&mut self, world: &mut World, entity: Entity) -> Result {
        let mut tfms = world.write_storage::<Transform>();
        let tfm = tfms
            .get_mut_or_default(entity)
            .ok_or_else(|| dead_entity_err(entity))?;
        tfm.position = Vec3::new(self.x, self.y, self.z);

        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
pub struct Name(pub String);

#[typetag::serde]
impl Recipe for Name {
    fn apply(&mut self, world: &mut World, entity: Entity) -> Result {
        let mut names = world.write_storage::<crate::common::Name>();
        names
            .insert(entity, crate::common::Name(self.0.clone()))
            .map_err(|e| specs_error(entity, e))?;
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
pub struct Camera;

#[typetag::serde]
impl Recipe for Camera {
    fn apply(&mut self, world: &mut World, entity: Entity) -> Result {
        let mut cameras = world.write_storage::<crate::camera::Camera>();
        cameras
            .insert(entity, crate::camera::Camera::default())
            .map_err(|error| specs_error(entity, error))?;
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct RecipesData(pub Vec<Vec<Box<dyn Recipe>>>);

pub struct Recipes {
    recipes: RecipesData,
    #[allow(unused)]
    debug_name: Option<String>,
}

pub fn load(world: &mut World, recipes: Recipes) -> std::result::Result<Vec<Entity>, RecipeError> {
    let mut entities = Vec::with_capacity(recipes.recipes.0.len());
    for list in recipes.recipes.0 {
        let entity = world.create_entity().build();
        for mut recipe in list {
            recipe.apply(world, entity)?;
        }
        entities.push(entity);
    }
    Ok(entities)
}

pub fn from_str(world: &mut World, ron: &str) -> Vec<Entity> {
    let recipes = ron::from_str::<RecipesData>(ron).unwrap();
    load(
        world,
        Recipes {
            recipes,
            debug_name: Some(ron.to_owned()),
        },
    )
    .unwrap()
}

pub fn to_str(recipes: Recipes) -> String {
    ron::to_string(&recipes.recipes).unwrap()
}

pub fn load_file<P>(world: &mut World, path: P) -> std::io::Result<Vec<Entity>>
where
    P: AsRef<Path>,
{
    let s = std::fs::read_to_string(path)?;
    Ok(from_str(world, &s))
}

#[cfg(test)]
mod test {
    use super::*;
    use specs::WorldExt;

    #[test]
    fn dbg() {
        let recipes = Recipes {
            recipes: RecipesData(vec![vec![
                Box::new(Pos {
                    x: 3.0,
                    y: 4.0,
                    z: 1.0,
                }),
                Box::new(Name(String::from("test"))),
                Box::new(Camera),
            ]]),
            debug_name: Some("test".to_owned()),
        };
        std::fs::write("test.ron", to_str(recipes)).unwrap()
    }

    // #[test]
    fn test_recipe_pos() {
        let input = "[
            [
                Pos{x: 0.0, y: 0.0, z: 0.0},
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
