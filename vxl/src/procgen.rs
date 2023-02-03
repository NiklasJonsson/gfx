use super::Chunk;

use rand::{RngCore, SeedableRng as _};

pub fn run(seed: [u8; 32]) -> Chunk {
    let mut chunk = Chunk {
        data: [0; Chunk::LEN as usize],
    };

    let mut rng = rand::rngs::SmallRng::from_seed(seed);
    for byte in chunk.data.iter_mut() {
        *byte = (rng.next_u32() & 0xFFFF) as u8;
    }

    chunk
}
