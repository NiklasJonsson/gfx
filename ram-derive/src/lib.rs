mod component;
mod visit;

#[proc_macro_derive(Component, attributes(component))]
pub fn component_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input).expect("Failed to parse derive input");
    proc_macro::TokenStream::from(component::impl_component(&ast))
}

#[proc_macro_derive(Visitable, attributes(visitable))]
pub fn visitable_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input).expect("Failed to parse derive input");
    proc_macro::TokenStream::from(visit::impl_visitable(&ast))
}
