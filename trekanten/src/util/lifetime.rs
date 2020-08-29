use std::marker::PhantomData;
use std::rc::Rc;

/// Runtime lifetime checking. A parent that wants to track the number of children that are alive
/// may create and hold one of these. A child that needs to outlive a parent holds a clone of the
/// parents, When the parent is dropped, it can check that the lifetime
/// token is unique. If it is, all children have been dropped already.
/// Uses Rc internally
pub struct LifetimeToken<T> {
    inner: Rc<u8>,
    _ty: PhantomData<T>,
}

impl<T> Clone for LifetimeToken<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
            _ty: PhantomData {},
        }
    }
}

impl<T> LifetimeToken<T> {
    pub fn is_unique(&self) -> bool {
        Rc::strong_count(&self.inner) == 1
    }

    pub fn new() -> Self {
        Self {
            inner: Rc::new(0u8),
            _ty: PhantomData {},
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for LifetimeToken<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LifetimeToken")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<T> Default for LifetimeToken<T> {
    fn default() -> Self {
        Self::new()
    }
}
