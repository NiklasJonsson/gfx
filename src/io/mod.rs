pub mod input;
pub mod windowing;

// Commands from runner thread to event thread
pub enum Command {
    Quit,
}

pub type EventQueue = crossbeam::queue::SegQueue<windowing::Event>;
pub type CommandQueue = std::sync::mpsc::Receiver<Command>;
