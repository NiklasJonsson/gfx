use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::DeriveInput;

struct CompileError {
    span: Span,
    msg: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.msg.fmt(f)
    }
}

fn derive_std140_compat(di: &DeriveInput) -> Result<TokenStream, CompileError> {
    let trait_name = quote! { trekanten::traits::Std140 };
    let name = &di.ident;

    let data = match &di.data {
        syn::Data::Struct(data) => data,
        _ => {
            return Err(CompileError {
                msg: String::from("Is not a struct. Only structs are supported for std140."),
                span: name.span(),
            });
        }
    };

    let (impl_generics, ty_generics, where_clause) = di.generics.split_for_impl();

    if data.fields.is_empty() {
        return Err(CompileError {
            msg: String::from("Has no members. Empty structs can't be std140 compatible."),
            span: name.span(),
        });
    }

    struct Acc {
        offset_expr: TokenStream,
        align_of_impls: TokenStream,
        max_alignment: TokenStream,
    }

    let start = Acc {
        offset_expr: quote! {},
        align_of_impls: quote! {},
        max_alignment: quote! { 0 },
    };

    let mk_field_fn_name = |field: &syn::Field, idx: usize| {
        field
            .ident
            .as_ref()
            .map(|field| {
                if field.to_string().starts_with('_') {
                    format_ident!("offset_of_field{}", field)
                } else {
                    format_ident!("offset_of_field_{}", field)
                }
            })
            .unwrap_or_else(|| format_ident!("offset_of_field_{}", idx))
    };

    let Acc {
        offset_expr,
        align_of_impls,
        max_alignment
    } = data.fields.iter().enumerate().fold(start, |acc, (i, f)| {
        let fn_name = mk_field_fn_name(f, i);
        let ty = &f.ty;
        let Acc {
            offset_expr: prev_end,
            align_of_impls,
            max_alignment,
        } = acc;

        let offset = if i == 0 {
            quote! { 0 }} else {
            quote!{ trekanten::util::round_to_multiple(#prev_end, <#ty as #trait_name>::ALIGNMENT) }
            };

        let align_of_impls = quote! {
                #align_of_impls

                pub const fn #fn_name() -> usize {
                    #offset
                }
        };

        let offset_expr = quote! {
                (Self::#fn_name() + <#ty as #trait_name>::SIZE)
        };

        let max_alignment = quote! {
            trekanten::util::max(#max_alignment, <#ty as #trait_name>::ALIGNMENT)
        };

        Acc {
            offset_expr,
            align_of_impls,
            max_alignment,
        }
    });

    Ok(quote! {
        unsafe impl #impl_generics #trait_name for #name #ty_generics #where_clause {
            // If the struct is not a member, we can get away without the padding, but otherwise we need it.
            // It doesn't hurt to add it in the general case.
            const SIZE: usize = trekanten::util::round_to_multiple(#offset_expr, Self::ALIGNMENT);
            const ALIGNMENT: usize = trekanten::util::round_to_multiple(#max_alignment, 16);
        }

        impl #impl_generics #name #ty_generics #where_clause {
            #align_of_impls
        }
    })
}

pub(crate) fn impl_derive(di: &DeriveInput) -> TokenStream {
    derive_std140_compat(di).unwrap_or_else(|err| {
        let msg = err.msg;
        quote::quote_spanned! {err.span=> compile_error!("{}", #msg) }
    })
}
