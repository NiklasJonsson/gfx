use std::time::{Duration, Instant};

#[derive(Default, Debug, Copy, Clone)]
pub struct DeltaTime(Duration);

impl DeltaTime {
    pub fn zero() -> DeltaTime {
        DeltaTime(Duration::new(0, 0))
    }

    pub fn as_ms(&self) -> f32 {
        1000.0 * self.as_secs()
    }

    pub fn as_secs(&self) -> f32 {
        self.0.as_secs_f32()
    }

    pub fn as_fps(&self) -> f32 {
        1.0 / self.0.as_secs_f32()
    }
}

impl From<DeltaTime> for Duration {
    fn from(dt: DeltaTime) -> Duration {
        dt.0
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
        self.0.as_secs_f32() * other
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
        let now = Instant::now();
        self.delta = DeltaTime(now - self.prev);
        self.prev = now;
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
