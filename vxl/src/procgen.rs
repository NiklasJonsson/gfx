use super::Chunk;

use rand::{Rng, RngCore, SeedableRng as _};

pub fn run(seed: [u8; 32]) -> Chunk {
    let mut chunk = Chunk {
        data: [0; Chunk::LEN as usize],
    };

    let mut rng = rand::rngs::SmallRng::from_seed(seed);
    let mut empty = 0;
    for byte in chunk.data.iter_mut() {
        *byte = rng.gen_bool(0.33) as u8;
        empty += (*byte == 0) as u32;
    }

    println!(
        "Generated chunk with: empty: {empty}, which is {}% out of all blocks",
        (empty as f32 / Chunk::LEN as f32) * 100.0
    );

    chunk
}
