use specs::prelude::*;
use specs::Component;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Component)]
#[storage(DenseVecStorage)]
pub struct Name(pub String);

impl<S> From<S> for Name
where
    String: From<S>,
{
    fn from(s: S) -> Self {
        Self(String::from(s))
    }
}

pub fn register_components(w: &mut World) {
    w.register::<Name>();
}
