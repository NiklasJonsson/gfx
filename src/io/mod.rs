pub mod input;
pub mod windowing;

pub type EventQueue = crossbeam::queue::SegQueue<windowing::Event>;
