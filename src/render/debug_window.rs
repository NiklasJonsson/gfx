use crate::common::Name;
use crate::ecs::prelude::*;
use crate::io::input::{ActionId, InputContext, InputContextError, KeyCode, MappedInput};
use crate::render;

use crate::editor;
use editor::Inspect as _;
use ramneryd_derive::Inspect;

use num_derive::FromPrimitive;

#[derive(Debug, Copy, Clone, PartialEq, Eq, FromPrimitive, Inspect)]
pub enum RenderMode {
    Opaque,
    Wireframe,
}

#[derive(Debug, Copy, Clone, PartialEq, Inspect)]
pub struct RenderSettings {
    // Affects all entities
    pub render_mode: RenderMode,
    pub render_bounding_box: bool,
    pub reload_shaders: bool,
    pub render_light_volumes: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            render_mode: RenderMode::Opaque,
            render_bounding_box: false,
            reload_shaders: false,
            render_light_volumes: false,
        }
    }
}

impl RenderSettings {
    pub const ID: &'static str = "RenderSettings";
}

fn get_input_context() -> Result<InputContext, InputContextError> {
    Ok(InputContext::builder(RenderSettings::ID)
        .description("Input for changing render settings")
        .with_action(KeyCode::O, RENDER_MODE_SWITCH)?
        .with_action(KeyCode::P, RENDER_BOUNDING_BOX_SWITCH)?
        .with_action(KeyCode::R, RELOAD_SHADERS)?
        .build())
}

const RENDER_MODE_SWITCH: ActionId = ActionId(0);
const RENDER_BOUNDING_BOX_SWITCH: ActionId = ActionId(1);
const RELOAD_SHADERS: ActionId = ActionId(2);

struct RenderSettingsSys {
    input_entity: Option<specs::Entity>,
}

impl<'a> System<'a> for RenderSettingsSys {
    type SystemData = (Write<'a, RenderSettings>, WriteStorage<'a, MappedInput>);

    fn run(&mut self, (mut r_settings, mut inputs): Self::SystemData) {
        use crate::io::input::Input;
        log::trace!("RenderSettingsSys: run");
        let inp = inputs
            .get_mut(self.input_entity.unwrap())
            .expect("Failed to get mapped input for RenderSettingsSys");

        for i in inp.iter() {
            match i {
                Input::Action(RENDER_MODE_SWITCH) => {
                    log::debug!("Render mode switch!");
                    r_settings.render_mode = match r_settings.render_mode {
                        RenderMode::Opaque => RenderMode::Wireframe,
                        RenderMode::Wireframe => RenderMode::Opaque,
                    };
                }
                Input::Action(RENDER_BOUNDING_BOX_SWITCH) => {
                    log::debug!("Render bounding box switch!");
                    r_settings.render_bounding_box = !r_settings.render_bounding_box;
                }
                Input::Action(RELOAD_SHADERS) => {
                    log::debug!("Reload runtime shaders!");
                    r_settings.reload_shaders = true;
                }
                i => unreachable!("{:?}", i),
            }
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        world.insert(RenderSettings::default());
        let ctx = get_input_context().expect("Failed to build settings input context");
        self.input_entity = Some(
            world
                .create_entity()
                .with(ctx)
                .with(Name::from(RenderSettings::ID))
                .build(),
        );
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder
        .with(
            RenderSettingsSys { input_entity: None },
            RenderSettings::ID,
            &[],
        )
        .with(ApplySettings, ApplySettings::ID, &[RenderSettings::ID])
}

pub(crate) fn build_ui<'a>(world: &mut World, ui: &imgui::Ui<'a>, pos: [f32; 2]) -> [f32; 2] {
    let mut settings = world.write_resource::<RenderSettings>();

    let size = [300.0, 85.0];

    imgui::Window::new(imgui::im_str!("Render debug"))
        .position(pos, imgui::Condition::FirstUseEver)
        .size(size, imgui::Condition::FirstUseEver)
        .build(&ui, || {
            settings.inspect_mut(ui, "");
            ui.text("Lights");
            let mut lights = world.write_storage::<render::light::Light>();
            let mut transforms = world.write_storage::<crate::math::Transform>();
            for (i, (light, tfm)) in (&mut lights, &mut transforms).join().enumerate() {
                let id = format!("Light##{}", i);
                editor::inspect::inspect_struct(
                    "",
                    Some(&id),
                    ui,
                    Some(|| {
                        tfm.inspect_mut(ui, "transform");
                        light.inspect_mut(ui, "light");
                    }),
                );
            }
        });

    size
}

pub struct ApplySettings;

impl ApplySettings {
    pub const ID: &'static str = "ApplySettings";
}

impl<'a> System<'a> for ApplySettings {
    type SystemData = (
        Write<'a, RenderSettings>,
        Entities<'a>,
        ReadStorage<'a, render::material::Material>,
        WriteStorage<'a, render::ReloadMaterial>,
        ReadStorage<'a, crate::math::BoundingBox>,
        WriteStorage<'a, render::bounding_box::DoRenderBoundingBox>,
        ReadStorage<'a, render::light::Light>,
        WriteStorage<'a, render::light::RenderLightVolume>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (
            mut render_settings,
            entities,
            materials,
            mut reload_materials,
            bounding_boxes,
            mut render_bbox,
            lights,
            mut render_light_cmds,
        ) = data;
        if render_settings.reload_shaders {
            for (ent, _mat) in (&entities, &materials).join() {
                reload_materials
                    .insert(ent, render::ReloadMaterial)
                    .expect("Failed to insert");
            }

            render_settings.reload_shaders = false;
        }

        if render_settings.render_bounding_box {
            for (ent, _bbox) in (&entities, &bounding_boxes).join() {
                render_bbox
                    .insert(ent, render::bounding_box::DoRenderBoundingBox)
                    .expect("Failed to insert");
            }

            render_settings.render_bounding_box = false;
        }

        if render_settings.render_light_volumes {
            for (ent, _light) in (&entities, &lights).join() {
                render_light_cmds
                    .insert(ent, render::light::RenderLightVolume)
                    .expect("Failed to insert");
            }

            render_settings.render_light_volumes = false;
        }
    }
}
