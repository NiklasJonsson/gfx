pub mod event;
pub mod input;

use winit::window::Window;

use specs::prelude::*;

// Commands from runner thread to event thread
pub enum Command {
    Quit,
}

pub type EventQueue = crossbeam::queue::SegQueue<event::Event>;
pub type CommandQueue = std::sync::mpsc::Receiver<Command>;

pub struct MainWindow {
    window: Window,
}

pub fn window_extents(window: &winit::window::Window) -> trekanten::util::Extent2D {
    let winit::dpi::PhysicalSize { width, height } = window.inner_size();
    trekanten::util::Extent2D { width, height }
}

impl MainWindow {
    pub fn cursor_grab(&mut self, cursor_grab: bool) {
        self.window
            .set_cursor_grab(cursor_grab)
            .expect("Unable to grab cursor");
        self.window.set_cursor_visible(!cursor_grab);
    }

    pub fn extents(&self) -> trekanten::util::Extent2D {
        window_extents(&self.window)
    }
}

pub fn setup(world: &mut World, window: winit::window::Window) {
    world.insert(input::CurrentFrameExternalInputs(Vec::new()));
    world.insert(MainWindow { window });
}
