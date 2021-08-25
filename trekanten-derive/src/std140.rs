use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::DeriveInput;

pub(crate) fn impl_derive(di: &DeriveInput) -> TokenStream {
    let trait_name = quote! { trekanten::traits::Std140 };
    let name = &di.ident;

    let data = match &di.data {
        syn::Data::Struct(data) => data,
        _ => {
            return quote::quote_spanned! {name.span()=>
                compile_error!("Is not a struct. Only structs are supported for std140.");
            };
        }
    };

    let (impl_generics, ty_generics, where_clause) = di.generics.split_for_impl();

    if data.fields.is_empty() {
        return quote::quote_spanned! {name.span()=>
            compile_error!("Has no members. Empty structs can't be std140 compatible.");
        };
    }

    struct Acc {
        offset_expr: TokenStream,
        align_of_impls: TokenStream,
    }

    let start = Acc {
        offset_expr: quote! { 0 },
        align_of_impls: quote! {},
    };

    let Acc {
        offset_expr,
        align_of_impls,
    } = data.fields.iter().enumerate().fold(start, |acc, (i, f)| {
        let fn_name = f
            .ident
            .as_ref()
            .map(|field| {
                if field.to_string().starts_with('_') {
                    format_ident!("alignment_of_field{}", field)
                } else {
                    format_ident!("alignment_of_field_{}", field)
                }
            })
            .unwrap_or_else(|| format_ident!("alignment_of_field_{}", i));
        let ty = &f.ty;
        let Acc {
            offset_expr,
            align_of_impls,
        } = acc;

        let start_offset =
            quote! { #offset_expr + ( #offset_expr % <#ty as #trait_name>::ALIGNMENT ) };

        Acc {
            offset_expr: quote! {
                #start_offset + <#ty as #trait_name>::SIZE
            },
            align_of_impls: quote! {
                #align_of_impls

                pub fn #fn_name() -> usize {
                    #start_offset
                }
            },
        }
    });

    quote! {
        unsafe impl #impl_generics #trait_name for #name #ty_generics #where_clause {
            const SIZE: usize = #offset_expr;
            // TODO: nested structs
            const ALIGNMENT: usize = 0;
        }

        impl #impl_generics #name #ty_generics #where_clause {
            #align_of_impls
        }
    }
}
