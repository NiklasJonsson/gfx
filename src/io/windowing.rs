use super::input;

use winit::event::DeviceEvent;
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

    pub fn collect_event<'a>(
        &mut self,
        event: winit::event::Event<'a, ()>,
    ) -> super::EventLoopControl {
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
                }
                WindowEvent::Focused(false) => {
                    log::debug!("Window lost focus, ignoring input");
                    self.update_action(Event::Unfocus);
                }
                WindowEvent::Focused(true) => {
                    log::debug!("Window gained focus, accepting input");
                    self.update_action(Event::Focus);
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
                if let DeviceEvent::MouseMotion { delta: (x, y) } = inner_event {
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
                Event::Input(v) if v.is_empty() => super::EventLoopControl::Continue,
                x => super::EventLoopControl::Done(r),
            }
        } else {
            super::EventLoopControl::Continue
        }
    }

    fn resolve(&mut self) -> Event {
        let new = match &self.action {
            Event::Input(_) => Event::Input(Vec::new()),
            Event::Focus => Event::Focus,
            Event::Unfocus => Event::Unfocus,
            Event::Quit => Event::Quit,
        };

        std::mem::replace(&mut self.action, new)
    }
}
