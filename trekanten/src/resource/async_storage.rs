use super::Handle;
use std::marker::PhantomData;

pub struct AsyncStorage<R> {
    _ty: PhantomData<R>,
}

impl<R> AsyncStorage<R> {
    pub fn allocate(&self) -> Handle<R> {
        todo!()
    }

    pub fn insert(&self, _h: Handle<R>, _data: R) {
        todo!();
    }
}
