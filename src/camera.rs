mod camera {

    #[derive(Debug)]
    pub struct Camera {
        position: [f32; 3],
        direction: [f32; 3],
        up: [f32; 3],
    }

    impl Camera {
        pub new(position: [f32;3}, direction: [f32; 3]) -> Self {
            Camera{position, direction, [0.0, 1.0, 0.0]}
        }

        pub move(&mut self, delta: &[f32; 3]) {
            self.position += delta;
        }
    }
}
