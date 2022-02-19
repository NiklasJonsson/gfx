use ramneryd::math;
use ramneryd::math::{Quat, Rgb, Rgba, Vec3};
use ramneryd::render;

use ramneryd::common::Name;
use ramneryd::ecs::prelude::*;
use ramneryd::{Module, ModuleLoader};

use clap::Parser;

use std::path::PathBuf;

#[derive(Debug, Parser)]
#[clap(author, version, about = "ramneryd debug binary")]
struct Args {
    #[clap(parse(from_os_str), name = "gltf-file", long)]
    gltf_files: Vec<PathBuf>,
    #[clap(parse(from_os_str), name = "rsf-file", long)]
    rsf_files: Vec<PathBuf>,
    #[clap(long)]
    spawn_plane: bool,
    #[clap(long)]
    spawn_cube: bool,
}

impl Module for Args {
    fn load(&mut self, loader: &mut ModuleLoader) {
        let world = &mut loader.world;

        self.gltf_files
            .iter()
            .for_each(|f| ramneryd::asset::gltf::load_asset(world, f));
        self.rsf_files
            .iter()
            .for_each(|f| ramneryd::asset::rsf::load_asset(world, f));

        let plane_side = 100.0;
        let plane_height = 1.0;
        if self.spawn_plane {
            world
                .create_entity()
                .with(Name::from("Plane"))
                .with(math::Transform::pos(0.0, -plane_height / 2.0, 0.0))
                .with(render::Shape::Box {
                    width: plane_side,
                    height: plane_height,
                    depth: plane_side,
                })
                .with(render::material::PhysicallyBased {
                    base_color_factor: Rgba {
                        r: 0.3,
                        g: 0.3,
                        b: 0.3,
                        a: 1.0,
                    },
                    metallic_factor: 0.0,
                    roughness_factor: 0.7,
                    ..Default::default()
                })
                .build();
        }

        if self.spawn_cube {
            world
                .create_entity()
                .with(Name::from("Cube"))
                .with(math::Transform::pos(0.0, 3.0, 0.0))
                .with(render::Shape::Box {
                    width: 1.0,
                    height: 1.0,
                    depth: 1.0,
                })
                .with(render::material::PhysicallyBased {
                    base_color_factor: Rgba {
                        r: 0.3,
                        g: 0.6,
                        b: 0.3,
                        a: 1.0,
                    },
                    metallic_factor: 0.0,
                    roughness_factor: 0.7,
                    ..Default::default()
                })
                .build();
        }
    }
}

#[derive(Component)]
struct Sun {
    angular_velocity: f32,
    angle: f32,
}

const SUN_HEIGHT: f32 = 10.0;

struct SunSimulation;

struct SunMovement;

impl SunMovement {
    const ID: &'static str = "SunMovement";
}

impl<'a> System<'a> for SunMovement {
    type SystemData = (
        WriteStorage<'a, Sun>,
        WriteStorage<'a, math::Transform>,
        ReadExpect<'a, ramneryd::Time>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut suns, mut transforms, time) = data;
        for (sun, tfm) in (&mut suns, &mut transforms).join() {
            let delta = time.delta_sim().as_secs();
            sun.angle += delta * sun.angular_velocity;

            let y = sun.angle.sin() * SUN_HEIGHT;
            let x = sun.angle.cos() * SUN_HEIGHT;

            if y < 0.0 {
                sun.angular_velocity *= -1.0;
            }

            tfm.position.x = x;
            tfm.position.y = y;

            tfm.rotation = Quat::rotation_from_to_3d(
                ramneryd::render::Light::DEFAULT_FACING,
                Vec3::from(0.0) - tfm.position,
            );
        }
    }
}

impl Module for SunSimulation {
    fn load(&mut self, loader: &mut ModuleLoader) {
        loader.add_system(SunMovement, SunMovement::ID, &[]);
        loader
            .world
            .create_entity()
            .with(Name::from("Sun"))
            .with(math::Transform {
                position: Vec3 {
                    x: 0.0,
                    y: SUN_HEIGHT,
                    z: 0.0,
                },
                rotation: Quat::rotation_from_to_3d(
                    ramneryd::render::Light::DEFAULT_FACING,
                    Vec3::new(0.0, -1.0, 0.0),
                ),
                scale: 1.0,
            })
            .with(Sun {
                angular_velocity: (3.0_f32).to_radians(),
                angle: std::f32::consts::FRAC_PI_2,
            })
            .with(ramneryd::render::Light::Directional {
                color: Rgb {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                },
            })
            .build();
    }
}

fn main() {
    ramneryd::Init::new()
        .with_module(Args::parse())
        .with_module(SunSimulation)
        .run();
}
