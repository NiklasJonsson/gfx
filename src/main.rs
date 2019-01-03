extern crate vulkano;
extern crate winit;

use winit::{Event, ControlFlow, WindowEvent};
use winit::dpi::LogicalSize;

struct App {
    events_loop: winit::EventsLoop,
    window: winit::Window,
}

impl App {
    fn main_loop(&mut self) {
        println!("main_loop()");

        let mut quit = false;

        while !quit {
            self.events_loop.poll_events(|event| {
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => quit = true,
                    _ => ()
                }
            });
        }
    }

    fn run(&mut self) {
        println!("run()");
        self.main_loop();
    }

    fn new() -> App {
        println!("new()");
        let events_loop = winit::EventsLoop::new();
        let window = winit::Window::new(&events_loop).unwrap();
        return App{events_loop, window};
    }
}

fn main() {
    let mut app = App::new();
    app.run();
}
