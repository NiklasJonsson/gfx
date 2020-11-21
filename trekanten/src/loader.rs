use crate::command::CommandBuffer;
use crate::device::Device;
use crate::resource::{AsyncStorage, Handle};

use std::sync::Arc;

pub trait ResourceCmd {
    fn start(&mut self, device: &Device, cmd_buf: &mut CommandBuffer);
    fn end(self);
}

pub trait Descriptor<R>: Clone {
    fn load(&self, handle: Handle<R>, storage: Arc<AsyncStorage<R>>) -> Box<dyn ResourceCmd>;
}

#[derive(Debug, Default, Clone)]
pub struct Loader {}

impl Loader {
    pub fn load<R, H>(&self, descriptor: impl Descriptor<R>, storage: Arc<AsyncStorage<R>>) -> H {
        let handle = storage.allocate();

        let _cmd = descriptor.load(handle, storage);
        todo!()
    }
}
