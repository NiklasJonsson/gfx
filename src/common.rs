use specs::prelude::*;
use specs::Component;

use crate::math::Position;

#[derive(Debug, Component)]
pub struct BoundingBox {
    pub min: Position,
    pub max: Position,
}

impl Default for BoundingBox {
    fn default() -> Self {
        BoundingBox {
            min: Position::max(),
            max: Position::min(),
        }
    }
}

impl BoundingBox {
    pub fn combine_with(&mut self, other: &Self) {
        // min
        let x = self.min.x().min(other.min.x());
        let y = self.min.y().min(other.min.y());
        let z = self.min.z().min(other.min.z());
        self.min = Position::new(x, y, z);

        // max
        let x = self.max.x().max(other.max.x());
        let y = self.max.y().max(other.max.y());
        let z = self.max.z().max(other.max.z());
        self.max = Position::new(x, y, z);
    }
}
