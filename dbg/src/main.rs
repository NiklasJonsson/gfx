use ram::math::{Quat, Rgb, Rgba, Transform, Vec3};
use ram::render;

use ram::common::Name;
use ram::ecs::prelude::*;
use ram::{Module, ModuleLoader};

use clap::Parser;

use std::path::PathBuf;

#[derive(Debug, Parser)]
#[clap(author, version, about = "ram debug binary")]
struct Args {
    #[clap(parse(from_os_str), name = "gltf-file", long)]
    gltf_files: Vec<PathBuf>,
    #[clap(parse(from_os_str), name = "rsf-file", long)]
    rsf_files: Vec<PathBuf>,
    #[clap(long)]
    spawn_plane: bool,
    #[clap(long)]
    spawn_cube: bool,
    #[clap(long)]
    many_cube_lights: bool,
    #[clap(long)]
    sun_simulation: bool,
}

struct Spawner {
    spawn_plane: bool,
    spawn_cube: bool,
    gltf_files: Vec<PathBuf>,
    rsf_files: Vec<PathBuf>,
}

impl Module for Spawner {
    fn load(&mut self, loader: &mut ModuleLoader) {
        let world = &mut loader.world;

        self.gltf_files
            .iter()
            .for_each(|f| ram::asset::gltf::load_asset(world, f));
        self.rsf_files
            .iter()
            .for_each(|f| ram::asset::rsf::load_asset(world, f));

        let plane_side = 100.0;
        let plane_height = 1.0;
        if self.spawn_plane {
            world
                .create_entity()
                .with(Name::from("Plane"))
                .with(Transform::pos(0.0, -plane_height / 2.0, 0.0))
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
                .with(Transform::pos(0.0, 3.0, 0.0))
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
        WriteStorage<'a, Transform>,
        ReadExpect<'a, ram::Time>,
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
                ram::render::Light::DEFAULT_FACING,
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
            .with(Transform {
                position: Vec3 {
                    x: 0.0,
                    y: SUN_HEIGHT,
                    z: 0.0,
                },
                rotation: Quat::rotation_from_to_3d(
                    ram::render::Light::DEFAULT_FACING,
                    Vec3::new(0.0, -1.0, 0.0),
                ),
                scale: 1.0,
            })
            .with(Sun {
                angular_velocity: (3.0_f32).to_radians(),
                angle: std::f32::consts::FRAC_PI_2,
            })
            .with(ram::render::Light::Directional {
                color: Rgb {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                },
            })
            .build();
    }
}

struct ManyCubeLights;

impl Module for ManyCubeLights {
    fn load(&mut self, loader: &mut ModuleLoader) {
        let world = &mut loader.world;

        let spawn_cube = |world: &mut World, i: usize, x: f32, z: f32| {
            world
                .create_entity()
                .with(Name::from(format!("Cube {i}")))
                .with(Transform::pos(x, 3.0, z))
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
        };
        let spawn_light = |world: &mut World, i: usize, x: f32, z: f32| {
            world
                .create_entity()
                .with(Name::from(format!("Spot light {i}")))
                .with(Transform {
                    position: Vec3 { x, y: 7.0, z },
                    // Pointing towards -y
                    rotation: Quat {
                        x: -0.70710677,
                        y: 0.0,
                        z: 0.0,
                        w: 0.70710677,
                    },
                    scale: 1.0,
                })
                .with(render::Light::Spot {
                    color: Rgb {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                    },
                    angle: std::f32::consts::FRAC_PI_8,
                    range: std::ops::Range {
                        start: 0.1,
                        end: 10.0,
                    },
                })
                .build();
        };
        const DIST: f32 = 5.0;

        let mut i = 0;
        let coords = [-1.0, 0.0, 1.0];
        for x in coords {
            for z in coords {
                let x = x * DIST;
                let z = z * DIST;
                spawn_cube(world, i, x, z);
                spawn_light(world, i, x, z);
                i += 1;
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    let mut init = ram::Init::new();

    init.add_module(Spawner {
        spawn_cube: args.spawn_cube,
        spawn_plane: args.spawn_plane,
        gltf_files: args.gltf_files,
        rsf_files: args.rsf_files,
    });

    if args.sun_simulation {
        init.add_module(SunSimulation);
    }

    if args.many_cube_lights {
        init.add_module(ManyCubeLights);
    }

    init.run();
}
