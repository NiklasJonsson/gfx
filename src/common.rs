use crate::ecs::prelude::*;

use serde::{Deserialize, Serialize};

#[derive(
    Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Component, Serialize, Deserialize,
)]
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
