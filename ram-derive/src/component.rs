use proc_macro2::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, DeriveInput, Lit, Meta, NestedMeta, Path};

pub(crate) fn impl_component(di: &DeriveInput) -> TokenStream {
    let name = &di.ident;
    let (impl_generics, ty_generics, where_clause) = di.generics.split_for_impl();

    let mut storage: Option<Path> = None;
    for attr in di.attrs.iter() {
        if attr.path.is_ident("component") {
            let meta = attr.parse_meta();
            match meta {
                Err(e) => {
                    let msg = format!("Failed to parse component attributes: {}", e);
                    let msg_ref: &str = &msg;
                    return quote::quote_spanned! {attr.span()=>
                        compile_error!(#msg_ref);
                    };
                }
                Ok(Meta::List(list)) => {
                    for nm in list.nested {
                        match nm {
                            NestedMeta::Meta(Meta::NameValue(nv)) => {
                                if nv.path.is_ident("storage") {
                                    if let Lit::Str(lit) = nv.lit {
                                        let p: Path = lit.parse().unwrap();
                                        storage = Some(p);
                                    }
                                }
                            }
                            _ => unimplemented!(),
                        }
                    }
                }
                _ => unimplemented!(),
            }
        }
    }

    let storage = storage.unwrap_or_else(|| syn::parse_quote!(DenseVecStorage));

    quote! {
        /// specs
        impl #impl_generics ::ram::ecs::prelude::Component for #name #ty_generics #where_clause {
            type Storage = ::ram::ecs::prelude::#storage<Self>;
        }
    }
}
