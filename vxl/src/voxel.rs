const N: u32 = 16;

pub struct Chunk {
    pub data: [u8; N as usize * N as usize * N as usize],
}

impl Chunk {
    pub const SIDE: u32 = N;
    pub const LEN: u32 = N * N * N;

    pub fn index(&self, x: u32, y: u32, z: u32) -> u32 {
        x + Self::SIDE * y + Self::SIDE * Self::SIDE * z
    }
}
