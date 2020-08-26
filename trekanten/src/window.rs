use crate::util;
use std::time::Duration;

pub trait Window {
    fn extents(&self) -> util::Extent2D;
}

pub const WINDOW_HEIGHT: u32 = 300;
pub const WINDOW_WIDTH: u32 = 300;
const WINDOW_TITLE: &str = "Trekanten";

pub type GlfwWindowEvents = std::sync::mpsc::Receiver<(f64, glfw::WindowEvent)>;

pub struct GlfwWindow {
    pub glfw: glfw::Glfw,
    pub window: glfw::Window,
    pub events: GlfwWindowEvents,
    frame_times: [std::time::Duration; 10],
    frame_time_idx: usize,
}

impl GlfwWindow {
    pub fn new() -> Self {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).expect("Failed to init glfw");
        assert!(glfw.vulkan_supported(), "No vulkan!");

        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

        let (mut window, events) = glfw
            .create_window(
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                WINDOW_TITLE,
                glfw::WindowMode::Windowed,
            )
            .expect("Failed to create GLFW window.");

        window.set_key_polling(true);

        Self {
            glfw,
            window,
            events,
            frame_times: [std::time::Duration::default(); 10],
            frame_time_idx: 0,
        }
    }

    pub fn set_frame_ms(&mut self, time: Duration) {
        self.frame_times[self.frame_time_idx] = time;

        if self.frame_time_idx == self.frame_times.len() - 1 {
            let avg = self
                .frame_times
                .iter()
                .fold(Duration::from_secs(0), |acc, &t| acc + t)
                / self.frame_times.len() as u32;
            let s = format!(
                "Trekanten (FPS: {:.2}, {:.2} ms)",
                1.0 / avg.as_secs_f32(),
                1000.0 * avg.as_secs_f32()
            );
            self.window.set_title(&s);
            self.frame_time_idx = 0;
        } else {
            self.frame_time_idx += 1;
        }
    }
}

impl Window for GlfwWindow {
    fn extents(&self) -> util::Extent2D {
        let (w, h) = self.window.get_framebuffer_size();
        util::Extent2D {
            width: w as u32,
            height: h as u32,
        }
    }
}

unsafe impl raw_window_handle::HasRawWindowHandle for GlfwWindow {
    fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
        self.window.raw_window_handle()
    }
}
