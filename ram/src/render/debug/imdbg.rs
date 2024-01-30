use crate::render::imgui::UIModule;

struct Entry {
    expr: &'static str,
    file: &'static str,
    line: u32,
    data: String,
}

struct ImDebug {
    entries: Vec<Entry>,
}

impl ImDebug {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

static IM_DEBUGGER: std::sync::Mutex<ImDebug> = std::sync::Mutex::new(ImDebug::new());

pub fn add_imdebug<T>(t: &T, expr: &'static str, file: &'static str, line: u32)
where
    T: std::fmt::Debug,
{
    let data = format!("{t:#?}");
    let mut guard = IM_DEBUGGER.lock().expect("Failed to lock imdbg mutex");
    guard.entries.push(Entry {
        expr,
        file,
        line,
        data,
    })
}

struct UIIntegration {
    swap: Vec<Entry>,
}

impl UIModule for UIIntegration {
    fn draw(&mut self, _: &mut specs::prelude::World, frame: &crate::render::imgui::UiFrame) {
        {
            self.swap.clear();
            let mut guard = IM_DEBUGGER.lock().expect("Failed to ");
            std::mem::swap(&mut self.swap, &mut guard.entries);
        }

        let ui = frame.inner();
        ui.window("ImDebug").build(|| {
            for (idx, entry) in self.swap.iter().enumerate() {
                if let Some(_tree_node) = ui.tree_node(&format!(
                    "[{}:{}] {}##{}",
                    entry.file, entry.line, entry.expr, idx
                )) {
                    ui.text(&entry.data);
                }
            }
        });
    }
}

/// Similar to dbg!(), this is immediate-debug, imdbg. It dumps the expression in an
/// imgui window. The implementation is coped from the std lib dbg and modified.
#[macro_export]
macro_rules! imdbg {
    // NOTE: We cannot use `concat!` to make a static string as a format argument
    // of `eprintln!` because `file!` could contain a `{` or
    // `$val` expression could be a block (`{ .. }`), in which case the `eprintln!`
    // will be malformed.
    () => {
        $crate::render::debug::imdbg::add_imdebug("", "", std::file!(), std::line!())
    };
    ($val:expr $(,)?) => {
        // Use of `match` here is intentional because it affects the lifetimes
        // of temporaries - https://stackoverflow.com/a/48732525/1063961
        match $val {
            tmp => {
                $crate::render::debug::imdbg::add_imdebug(&tmp, std::stringify!($val), std::file!(), std::line!());
                tmp
            }
        }
    };
    ($($val:expr),+ $(,)?) => {
        ($($crate::imdbg!($val)),+,)
    };
}

pub fn ui_module() -> Box<dyn UIModule> {
    Box::new(UIIntegration { swap: vec![] })
}
