use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use trekanten::util;
use trekanten::{buffer, BufferMutability};

use trekanten::ResourceManager as _;

use std::time::{Duration, Instant};

const WINDOW_HEIGHT: u32 = 300;
const WINDOW_WIDTH: u32 = 300;
const WINDOW_TITLE: &str = "Trekanten Vulkan Tutoria";

struct State {
    pub window: winit::window::Window,
    frame_times: [Duration; 10],
    frame_time_idx: usize,
    start: Instant,
    frame_start: Instant,
}

impl State {
    pub fn new() -> (Self, EventLoop<()>) {
        let ev = EventLoop::new();
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .with_title(WINDOW_TITLE)
            .build(&ev)
            .expect("Failed to build window");

        (
            Self {
                window,
                frame_times: [std::time::Duration::default(); 10],
                frame_time_idx: 0,
                start: Instant::now(),
                frame_start: Instant::now(),
            },
            ev,
        )
    }

    pub fn start_frame(&mut self) {
        self.frame_start = std::time::Instant::now();
    }

    pub fn end_frame(&mut self) {
        let time = std::time::Instant::now() - self.frame_start;
        self.frame_times[self.frame_time_idx] = time;

        if self.frame_time_idx < self.frame_times.len() - 1 {
            self.frame_time_idx += 1;
            return;
        }

        debug_assert!(self.frame_time_idx == (self.frame_times.len() - 1));

        let avg = self
            .frame_times
            .iter()
            .fold(Duration::from_secs(0), |acc, &t| acc + t)
            / self.frame_times.len() as u32;
        let s = format!(
            "{} (FPS: {:.2}, {:.2} ms)",
            WINDOW_TITLE,
            1.0 / avg.as_secs_f32(),
            1000.0 * avg.as_secs_f32()
        );
        self.window.set_title(&s);
        self.frame_time_idx = 0;
    }

    fn extents(&self) -> util::Extent2D {
        let PhysicalSize { width, height } = self.window.inner_size();
        util::Extent2D { width, height }
    }
}

fn main() {
    env_logger::init();

    let (mut state, event_loop) = State::new();
    let mut rendering =
        vxl::Rendering::new(&state.window, state.extents()).expect("Failed to init");

    let chunk = vxl::procgen::run([0; 32]);
    let vxl::Mesh { vertices, indices } = vxl::meshing::mesh(&chunk);
    let vertices = rendering
        .renderer
        .create_resource_blocking(trekanten::buffer::VertexBufferDescriptor::from_vec(
            vertices,
            BufferMutability::Immutable,
        ))
        .expect("Failed to init vbuf");
    let indices = rendering
        .renderer
        .create_resource_blocking(trekanten::buffer::IndexBufferDescriptor::from_vec(
            indices,
            BufferMutability::Immutable,
        ))
        .expect("Failed to init ibuf");
    let draw_commands = [vxl::RenderCmd { vertices, indices }];

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => {
                state.start_frame();
                rendering.render(state.extents(), &draw_commands);
                state.end_frame();
            }
            _ => (),
        }
    });
}
