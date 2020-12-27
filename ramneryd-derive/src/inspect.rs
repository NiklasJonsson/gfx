use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{spanned::Spanned as _, DeriveInput, Fields, FieldsUnnamed};

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
fn inspect_enum_body(data: &syn::DataEnum, is_mut: bool) -> TokenStream {
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
            Fields::Named(_) => quote_spanned! {variant.span()=> {#(#fields_lhs),*}},
            Fields::Unnamed(_) => quote_spanned! {variant.span()=> (#(#fields_lhs),*)},
            Fields::Unit => quote! {},
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

            let field_name = match &f.ident {
                Some(i) => quote_spanned! {f.ident.span()=>#i},
                None => {
                    let i = syn::LitInt::new(format!("{}", i).as_str(), f.ident.span());
                    quote_spanned! {f.span()=> #i}
                }
            };

            quote_spanned! {f.span()=>
                <#ty as crate::editor::Inspect>::#fn_name(#field, ui, stringify!(#field_name));
            }
        });

        let leaf = if let Fields::Unit = variant.fields {
            quote! {true}
        } else {
            quote! {false}
        };

        let fields_rhs = quote! {
            let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), stringify!(#id));
            if imgui::CollapsingHeader::new(&ty)
                .default_open(true)
                .leaf(#leaf)
                .build(ui) {
                ui.indent();
                #(#fields_rhs)*
                ui.unindent();
            }
        };

        quote_spanned! {variant.span()=>
            Self::#id #fields_lhs => { #fields_rhs }
        }
    });

    quote! {
        match self {
            #(#variants_impl)*
        }
    }
}

fn inspect_enum(di: &DeriveInput) -> TokenStream {
    let name = &di.ident;

    let [body, body_mut] = match &di.data {
        syn::Data::Enum(data) => [
            inspect_enum_body(data, false),
            inspect_enum_body(data, true),
        ],
        _ => panic!("Internal error: should be enum"),
    };

    quote! {
        impl crate::editor::Inspect for #name {
            fn inspect<'a>(&self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                if !name.is_empty() {
                    ui.text(format!("{}:", name));
                    ui.same_line(0.0);
                }

                #body
            }

            fn inspect_mut<'a>(&mut self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                if !name.is_empty() {
                    ui.text(format!("{}:", name));
                    ui.same_line(0.0);
                }

                #body_mut
            }
        }
    }
}

fn inspect_struct_data(data: &syn::DataStruct, is_mut: bool) -> TokenStream {
    let fn_name = inspect_fn_name(is_mut);
    let maybe_mut = maybe_mut(is_mut);
    let one_field = data.fields.len() == 1;

    let fields = data.fields.iter().enumerate().map(|(i, f)| {
        let field = f.ident.as_ref().map(|x| quote! {#x}).unwrap_or_else(|| {
            let i = syn::Index::from(i);
            quote! {#i}
        });
        let ty = &f.ty;
        let name = if f.ident.is_none() && one_field {
            quote_spanned! {f.span()=> ""}
        } else {
            quote_spanned! {f.span()=> stringify!(#field)}
        };

        quote_spanned! {f.span()=>
            <#ty as crate::editor::Inspect>::#fn_name(&#maybe_mut self.#field, ui, #name);
        }
    });

    quote! {
        #(#fields)*
    }
}

fn inspect_struct(di: &DeriveInput) -> TokenStream {
    let name = &di.ident;

    let [body, body_mut] = match &di.data {
        syn::Data::Struct(data) => [
            inspect_struct_data(data, false),
            inspect_struct_data(data, true),
        ],
        _ => panic!("Internal error: should be struct"),
    };

    quote! {
        impl crate::editor::Inspect for #name {
            fn inspect<'a>(&self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                crate::editor::inspect::inspect_struct(name, Some(stringify!(#name)),
                    std::mem::size_of::<Self>() == 0, ui, || {
                    #body
                });
            }

            fn inspect_mut<'a>(&mut self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                crate::editor::inspect::inspect_struct(name, Some(stringify!(#name)),
                std::mem::size_of::<Self>() == 0, ui, || {
                    #body_mut
                });
            }
        }
    }
}
pub(crate) fn impl_imgui_inspect(di: &DeriveInput) -> TokenStream {
    let inspect_impl = match &di.data {
        syn::Data::Struct(_) => inspect_struct(di),
        syn::Data::Enum(_) => inspect_enum(di),
        _ => unimplemented!("Only Struct or enum is supported"),
    };

    quote! {
        #inspect_impl
    }
}
