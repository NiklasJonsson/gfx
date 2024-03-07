mod std140;

#[proc_macro_derive(Std140)]
pub fn std140_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = syn::parse(input).expect("Failed to parse derive input");
    proc_macro::TokenStream::from(std140::impl_derive(&ast))
}
