use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{spanned::Spanned as _, DeriveInput};

fn inspect_fn_name(is_mut: bool) -> syn::Ident {
    if is_mut {
        Ident::new("inspect_mut", Span::call_site())
    } else {
        Ident::new("inspect", Span::call_site())
    }
}

fn maybe_mut(is_mut: bool) -> TokenStream {
    let maybe_mut: Vec<syn::token::Mut> = if is_mut {
        vec![syn::token::Mut {
            span: Span::call_site(),
        }]
    } else {
        vec![]
    };

    quote! {#(#maybe_mut)*}
}

fn imgui_inspect_struct(data: &syn::DataStruct, is_mut: bool) -> TokenStream {
    let fn_name = inspect_fn_name(is_mut);
    let maybe_mut = maybe_mut(is_mut);

    let fields = data.fields.iter().enumerate().map(|(i, f)| {
        let field = f.ident.as_ref().map(|x| quote!{#x}).unwrap_or_else(|| {
            let i = syn::Index::from(i);
            quote!{#i}
        });
        let ty = &f.ty;
        quote_spanned! {f.span()=>
            <#ty as crate::editor::Inspect>::#fn_name(&#maybe_mut self.#field, ui, stringify!(#field));
        }
    });

    quote! {
        #(#fields)*
    }
}

fn imgui_inspect_enum(data: &syn::DataEnum, is_mut: bool) -> TokenStream {
    let fn_name = inspect_fn_name(is_mut);
    let maybe_mut = maybe_mut(is_mut);

    let variants_impl = data.variants.iter().map(|variant| {
        let id = &variant.ident;
        let fields_lhs = variant
            .fields
            .iter()
            .enumerate()
            .map(|(i, f)| match &f.ident {
                Some(i) => quote_spanned! {f.ident.span()=>ref #maybe_mut #i},
                None => {
                    let i = format_ident!("field_{}", i);
                    quote_spanned! {f.span()=>ref #maybe_mut #i}
                }
            });

        let fields_lhs = match variant.fields {
            syn::Fields::Named(_) => quote_spanned! {variant.span()=> {#(#fields_lhs),*}},
            syn::Fields::Unnamed(_) => quote_spanned! {variant.span()=> (#(#fields_lhs),*)},
            syn::Fields::Unit => quote_spanned! {variant.span()=> },
        };

        let fields_rhs = variant.fields.iter().enumerate().map(|(i, f)| {
            let ty = &f.ty;
            let field = match &f.ident {
                Some(i) => quote_spanned! {f.ident.span()=>#i},
                None => {
                    let i = format_ident!("field_{}", i);
                    quote_spanned! {f.span()=> #i}
                }
            };

            quote_spanned! {f.span()=>
                <#ty as crate::editor::Inspect>::#fn_name(#field, ui, stringify!(#field));
            }
        });

        let fields_rhs = quote! {
            #(#fields_rhs)*
        };

        quote_spanned! {variant.span()=>
            Self::#id #fields_lhs => {
                let name = imgui::ImString::new(stringify!(#id));
                if imgui::CollapsingHeader::new(&name)
                    .default_open(true)
                    .build(ui) {
                    ui.indent();
                    #fields_rhs
                    ui.unindent();
                }
            }
        }
    });

    quote! {
        match self {
            #(#variants_impl)*
        }
    }
}

pub(crate) fn impl_imgui_inspect(di: &DeriveInput) -> proc_macro2::TokenStream {
    let name = &di.ident;

    let [body, body_mut] = match &di.data {
        syn::Data::Struct(data) => [
            imgui_inspect_struct(data, false),
            imgui_inspect_struct(data, true),
        ],
        syn::Data::Enum(data) => [
            imgui_inspect_enum(data, false),
            imgui_inspect_enum(data, true),
        ],
        _ => unimplemented!("Only Struct or enum is supported"),
    };

    quote! {
        impl crate::editor::Inspect for #name {
            fn inspect<'a>(&self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                ui.indent();
                #body
                ui.unindent();
            }

            fn inspect_mut<'a>(&mut self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                ui.indent();
                #body_mut
                ui.unindent();
            }
        }
    }
}
