extern crate vulkano;
extern crate winit;

struct App {


}

impl App {
    fn setup_window(&mut self) {
        println!("setup_window()");
    }

    fn setup_vulkan(&mut self) {
        println!("setup_vulkan()");
    }

    fn main_loop(&mut self) {
        println!("main_loop()");
    }

    fn run(&mut self) {
        println!("run()");
        self.setup_window();
        self.setup_vulkan();
        self.main_loop();
    }

    fn new() -> App {
        return App{};
    }
}

fn main() {
    let mut app = App::new();
    app.run();
}
