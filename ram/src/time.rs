use std::time::{Duration, Instant};

#[derive(Debug, Copy, Clone)]
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

#[allow(dead_code)]
pub struct Time {
    delta: DeltaTime,
    prev: Instant,
    start: Instant,
}

impl Time {
    pub fn tick(&mut self) -> DeltaTime {
        let now = Instant::now();
        self.delta = DeltaTime(now - self.prev);
        self.prev = now;
        self.delta
    }

    #[allow(dead_code)]
    pub fn delta_sim(&self) -> DeltaTime {
        self.delta
    }

    #[allow(dead_code)]
    pub fn elapsed_real(&self) -> DeltaTime {
        DeltaTime(Instant::now() - self.start)
    }
}

impl Default for Time {
    fn default() -> Self {
        Self {
            delta: DeltaTime::zero(),
            prev: Instant::now(),
            start: Instant::now(),
        }
    }
}
