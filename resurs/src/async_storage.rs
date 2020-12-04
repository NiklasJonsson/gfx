use super::Handle;

pub enum Async<R> {
    Pending,
    Available(R),
}

impl<R: Clone> Clone for Async<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Pending => Self::Pending,
            Self::Available(r) => Self::Available(r.clone()),
        }
    }
}

impl<R: Copy> Copy for Async<R> {}

impl<R> std::fmt::Debug for Async<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Async<{}>", std::any::type_name::<R>())?;
        if self.is_pending() {
            write!(f, "Pending")
        } else {
            write!(f, "Available")
        }
    }
}

impl<R> Async<R> {
    pub fn as_ref(&self) -> Async<&R> {
        match self {
            Self::Pending => Async::<&R>::Pending,
            Self::Available(r) => Async::<&R>::Available(r),
        }
    }

    pub fn as_mut_ref(&mut self) -> Async<&mut R> {
        match self {
            Self::Pending => Async::<&mut R>::Pending,
            Self::Available(r) => Async::<&mut R>::Available(r),
        }
    }

    pub fn is_pending(&self) -> bool {
        match self {
            Self::Pending => true,
            _ => false,
        }
    }

    pub fn expect(self, msg: &str) -> R {
        match self {
            Self::Pending => panic!("{}", msg),
            Self::Available(r) => r,
        }
    }

    pub fn map<U, F>(self, f: F) -> Async<U>
    where
        F: FnOnce(R) -> U,
    {
        match self {
            Self::Pending => Async::<U>::Pending,
            Self::Available(r) => Async::<U>::Available(f(r)),
        }
    }
}

impl<R> Handle<Async<R>> {
    pub fn unwrap_async(self) -> Handle<R> {
        Handle::<R>::new(self.id)
    }
}

impl<R> Handle<R> {
    pub fn wrap_async(&self) -> Handle<Async<R>> {
        Handle::<Async<R>>::new(self.id)
    }
}
