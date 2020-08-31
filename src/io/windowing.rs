use super::input;

use std::sync::Arc;
use winit::event::ElementState;
use winit::event::WindowEvent;

// TODO: Handle resized here as well
#[derive(Debug)]
pub enum Event {
    Quit,
    Focus,
    Unfocus,
    Input(Vec<input::ExternalInput>),
}

impl Default for Event {
    fn default() -> Self {
        Event::Input(Vec::new())
    }
}

impl Event {
    fn update_with(self, new: Self) -> Self {
        use Event::*;
        match (new, self) {
            (_, Quit) => Quit,
            (Quit, _) => Quit,
            (Unfocus, _) => Unfocus,
            (Focus, Unfocus) => Focus,
            (_, Unfocus) => Unfocus,
            (Input(mut new), Input(mut old)) => Input({
                old.append(&mut new);
                old
            }),
            (Input(vec), Focus) => Input(vec),
            (Focus, Input(v)) => {
                log::warn!("Spurios focus event received, ignoring");
                Input(v)
            }
            // Sometimes there are several focus events in a row
            (Focus, Focus) => Focus,
        }
    }
}

#[derive(Debug)]
pub enum EventLoopControl {
    SendEvent(Event),
    Continue,
    Quit,
}

pub struct EventManager {
    action: Event,
}

impl EventManager {
    pub fn new() -> Self {
        Self {
            action: Event::Input(Vec::new()),
        }
    }

    fn update_action(&mut self, action: Event) {
        let cur = std::mem::replace(&mut self.action, Event::Quit);
        self.action = cur.update_with(action);
    }

    pub fn collect_event<'a>(&mut self, event: winit::event::Event<'a, ()>) -> EventLoopControl {
        log::trace!("Received event: {:?}", event);
        let mut resolve = false;
        use winit::event::Event as WinEvent;
        match event {
            WinEvent::WindowEvent {
                event: inner_event, ..
            } => match inner_event {
                WindowEvent::CloseRequested => {
                    log::debug!("Received CloseRequested window event");
                    self.update_action(Event::Quit);
                    resolve = true;
                }
                WindowEvent::Focused(false) => {
                    log::debug!("Window lost focus, ignoring input");
                    self.update_action(Event::Unfocus);
                    resolve = true;
                }
                WindowEvent::Focused(true) => {
                    log::debug!("Window gained focus, accepting input");
                    self.update_action(Event::Focus);
                    resolve = true;
                }
                WindowEvent::KeyboardInput {
                    device_id, input, ..
                } => {
                    log::debug!("Captured key: {:?} from {:?}", input, device_id);
                    let is_pressed = input.state == ElementState::Pressed;
                    if let Some(key) = input.virtual_keycode {
                        let ei = if is_pressed {
                            input::ExternalInput::KeyPress(key)
                        } else {
                            input::ExternalInput::KeyRelease(key)
                        };

                        self.update_action(Event::Input(vec![ei]));
                    } else {
                        log::warn!("Key clicked but no virtual key mapped!");
                    }
                }
                _ => log::trace!("... ignoring"),
            },
            WinEvent::DeviceEvent {
                event: inner_event, ..
            } => {
                if let winit::event::DeviceEvent::MouseMotion { delta: (x, y) } = inner_event {
                    log::debug!("Captured mouse motion: ({:?}, {:?})", x, y);
                    let ei = input::ExternalInput::MouseDelta { x, y };
                    self.update_action(Event::Input(vec![ei]));
                } else {
                    log::trace!("Ignoring device event {:?}", inner_event);
                }
            }
            WinEvent::MainEventsCleared | WinEvent::LoopDestroyed => {
                resolve = true;
            }
            e => log::debug!("Ignoring high level event {:?}", e),
        };

        if resolve {
            let r = self.resolve();
            match &r {
                Event::Input(v) if v.is_empty() => EventLoopControl::Continue,
                Event::Quit => EventLoopControl::Quit,
                _ => EventLoopControl::SendEvent(r),
            }
        } else {
            EventLoopControl::Continue
        }
    }

    fn resolve(&mut self) -> Event {
        let new = match &self.action {
            Event::Input(_) => Event::Input(Vec::new()),
            Event::Quit => Event::Quit,
            Event::Focus => Event::Focus,
            Event::Unfocus => Event::Unfocus,
        };

        std::mem::replace(&mut self.action, new)
    }
}

use std::sync::mpsc;

// This function signature is copied from
pub fn event_thread_work(
    event_manager: &mut EventManager,
    event_queue: Arc<super::EventQueue>,
    command_queue: &super::CommandQueue,
    winit_event: winit::event::Event<()>,
    control_flow: &mut winit::event_loop::ControlFlow,
) {
    match command_queue.try_recv() {
        Err(mpsc::TryRecvError::Empty) => (),
        Err(mpsc::TryRecvError::Disconnected) => {
            log::info!("Runner thread has disconnected, event thread exiting");
            *control_flow = winit::event_loop::ControlFlow::Exit;
            return;
        }
        Ok(super::Command::Quit) => {
            log::info!("Runner thread send quit command, event thread exiting");
            *control_flow = winit::event_loop::ControlFlow::Exit;
            return;
        }
    }

    // Since this is a separate thread, it is fine to wait
    *control_flow = winit::event_loop::ControlFlow::Wait;

    match event_manager.collect_event(winit_event) {
        EventLoopControl::SendEvent(event) => {
            log::debug!("Sending event on queue: {:?}", event);
            event_queue.push(event)
        }
        EventLoopControl::Continue => (),
        EventLoopControl::Quit => {
            log::info!("Event loop thread received quit");
            log::info!("Sending {:?} on event queue", Event::Quit);
            event_queue.push(Event::Quit);
            log::info!("Event loop thread exiting");
            *control_flow = winit::event_loop::ControlFlow::Exit;
        }
    }
}
