#[macro_export]
macro_rules! register_module_systems {
    ($executor:ident, $($mod:path),+) => {
        [
            // We can't use $mod::register_systems because it's not implemented:
            // https://github.com/rust-lang/rust/issues/48067
            // Instead, we import the module and use the alias name when constructing the function name
            // For some reason, without the 'as' cast, the compiler thinks each block is a separate type.
            // The cast is there to tell it all types are the same.
            $(
                { use $mod as base; base::register_systems } as fn(crate::ecs::ExecutorBuilder<'a, 'b>) -> crate::ecs::ExecutorBuilder<'a, 'b>,
            )+
        ]
        .iter()
        .fold($executor, |a, f| f(a))
    };
}
