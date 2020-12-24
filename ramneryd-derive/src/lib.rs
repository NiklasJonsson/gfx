mod component;
mod inspect;

#[proc_macro_derive(Inspect)]
pub fn inspect_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input).expect("Failed to parse derive input");
    proc_macro::TokenStream::from(inspect::impl_imgui_inspect(&ast))
}

#[proc_macro_derive(Component, attributes(storage, component))]
pub fn component_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input).expect("Failed to parse derive input");
    proc_macro::TokenStream::from(component::impl_component(&ast))
}
