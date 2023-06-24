pub mod event;
pub mod input;

use winit::window::Window;

pub use input::InputModule;

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

#[allow(dead_code)]
impl MainWindow {
    pub fn new(window: Window) -> Self {
        Self { window }
    }

    pub fn cursor_grab(&mut self, cursor_grab: bool) {
        let mode = if cursor_grab {
            winit::window::CursorGrabMode::Locked
        } else {
            winit::window::CursorGrabMode::None
        };
        self.window
            .set_cursor_grab(mode)
            .expect("Unable to grab cursor");
        self.window.set_cursor_visible(!cursor_grab);
    }

    pub fn extents(&self) -> trekanten::util::Extent2D {
        window_extents(&self.window)
    }
}
