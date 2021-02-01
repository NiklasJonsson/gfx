use trekanten::loader::ResourceLoader;
use trekanten::texture;
use trekanten::mem::UniformBuffer;
use trekanten::{BufferHandle, Handle, Loader};

use crate::ecs::prelude::*;
use ramneryd_derive::Inspect;

#[derive(Debug, Clone, Inspect)]
pub struct TextureUse {
    pub handle: Handle<texture::Texture>,
    pub coord_set: u32,
}

#[derive(Debug, Clone, Component)]
#[component(inspect)]
pub enum Material {
    Unlit {
        color_uniform: BufferHandle<UniformBuffer>,
    },
    PBR {
        material_uniforms: BufferHandle<UniformBuffer>,
        normal_map: Option<TextureUse>,
        base_color_texture: Option<TextureUse>,
        metallic_roughness_texture: Option<TextureUse>,
        has_vertex_colors: bool,
    },
}

// Option is here to handle WriteStorage::remove() not returning the contents
// TODO: Replace Option
#[derive(Debug, Clone, Component)]
#[component(inspect)]
pub struct PendingMaterial(Option<Material>);

impl From<Material> for PendingMaterial {
    fn from(m: Material) -> Self {
        Self(Some(m))
    }
}

struct ResolvePending;

impl ResolvePending {
    const ID: &'static str = "ResolvePending<Material>";
}

impl<'a> System<'a> for ResolvePending {
    type SystemData = (
        WriteStorage<'a, Material>,
        WriteStorage<'a, PendingMaterial>,
        Entities<'a>,
        ReadExpect<'a, Loader>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut materials, mut pending, entities, loader) = data;

        let mut done = specs::BitSet::new();
        for (ent, pend) in (&entities, &pending).join() {
            let d = match pend.0.as_ref().unwrap() {
                Material::Unlit { color_uniform } => {
                    loader.is_done(color_uniform).expect("Bad handle")
                }
                Material::PBR {
                    material_uniforms,
                    normal_map,
                    base_color_texture,
                    metallic_roughness_texture,
                    ..
                } => {
                    let tex_done = |tex: &Option<TextureUse>| -> bool {
                        tex.as_ref()
                            .map(|t| loader.is_done(&t.handle).expect("Bad handle"))
                            .unwrap_or(true)
                    };

                    tex_done(normal_map)
                        && tex_done(base_color_texture)
                        && tex_done(metallic_roughness_texture)
                        && loader.is_done(material_uniforms).expect("Bad handle")
                }
            };

            if d {
                done.add(ent.id());
            }
        }

        for (ent, _) in (&entities, done).join() {
            let mat = pending
                .get_mut(ent)
                .expect("bad bitset")
                .0
                .take()
                .expect("This should be available here");

            materials.insert(ent, mat).unwrap();
            pending.remove(ent);
        }
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(ResolvePending, ResolvePending::ID, &[])
}
