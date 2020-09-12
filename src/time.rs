use std::time::{Duration, Instant};

#[derive(Default, Debug, Copy, Clone)]
pub struct DeltaTime(Duration);

impl DeltaTime {
    pub fn zero() -> DeltaTime {
        DeltaTime(Duration::new(0, 0))
    }

    pub fn to_f32(self) -> f32 {
        (self.0.as_secs() as f64 + self.0.subsec_nanos() as f64 / 1_000_000_000.0) as f32
    }

    pub fn as_fps(self) -> f32 {
        1.0 / self.to_f32()
    }
}

impl Into<Duration> for DeltaTime {
    fn into(self) -> Duration {
        self.0
    }
}

impl From<Duration> for DeltaTime {
    fn from(dur: Duration) -> Self {
        DeltaTime(dur)
    }
}

impl std::ops::Mul<f32> for DeltaTime {
    type Output = f32;
    fn mul(self, other: f32) -> Self::Output {
        self.to_f32() * other
    }
}

pub struct Timer {
    delta: DeltaTime,
    prev: Instant,
    start: Instant,
}

impl Timer {
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    pub fn tick(&mut self) {
        self.delta = DeltaTime(self.prev.elapsed());
        self.prev = Instant::now();
    }

    pub fn delta(&self) -> DeltaTime {
        self.delta
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self {
            delta: DeltaTime::zero(),
            prev: Instant::now(),
            start: Instant::now(),
        }
    }
}