use crate::ecs::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Component)]
#[component(inspect)]
pub struct Name(pub String);

impl<S> From<S> for Name
where
    String: From<S>,
{
    fn from(s: S) -> Self {
        Self(String::from(s))
    }
}
