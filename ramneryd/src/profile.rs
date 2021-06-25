use crate::ecs::prelude::*;

mod puffin {
    pub use super::*;

    #[cfg(feature = "profile-with-puffin")]
    pub(super) fn build<'a>(world: &mut World, ui: &imgui::Ui<'a>) {
        let mut profiler_ui = world.write_resource::<puffin_imgui::ProfilerUi>();
        profiler_ui.window(ui);
    }

    #[cfg(not(feature = "profile-with-puffin"))]
    pub(super) fn build<'a>(_: &mut World, _: &imgui::Ui<'a>) {}

    #[cfg(feature = "profile-with-puffin")]
    pub(super) fn setup(world: &mut World) {
        world.insert(puffin_imgui::ProfilerUi::default());
    }

    #[cfg(not(feature = "profile-with-puffin"))]
    pub(super) fn setup(_: &mut World) {}
}

pub fn build_ui<'a>(world: &mut World, ui: &imgui::Ui<'a>, _pos: [f32; 2]) -> [f32; 2] {
    puffin::build(world, ui);

    [0.0, 0.0]
}

pub fn setup(world: &mut World) {
    puffin::setup(world);
}
