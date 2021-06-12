use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{
    parse_quote, spanned::Spanned as _, DeriveInput, Fields, GenericParam, Generics, Meta,
    NestedMeta,
};

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

fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(ref mut type_param) = *param {
            type_param.bounds.push(parse_quote!(crate::editor::Inspect));
        }
    }
    generics
}

fn inspect_enum_body(data: &syn::DataEnum, name: &Ident, is_mut: bool) -> TokenStream {
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
            let ty = imgui::im_str!("enum {}::{}", stringify!(#name), stringify!(#id));
            if imgui::CollapsingHeader::new(&ty)
                .default_open(!name.is_empty())
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
            inspect_enum_body(data, name, false),
            inspect_enum_body(data, name, true),
        ],
        _ => panic!("Internal error: should be enum"),
    };

    let generics = add_trait_bounds(di.generics.clone());
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics crate::editor::Inspect for #name #ty_generics #where_clause {
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
    let n_fields = data.fields.len();

    let fields = data.fields.iter().enumerate().map(|(i, f)| {
        // Ignore if needed
        for attr in f.attrs.iter() {
            if attr.path.is_ident("inspect") {
                match attr.parse_meta() {
                    Err(e) => {
                        let msg = format!("Failed to parse component attributes: {}", e);
                        let msg_ref: &str = &msg;
                        return quote::quote_spanned! {attr.span()=>
                            compile_error!(#msg_ref);
                        };
                    }
                    Ok(Meta::List(list)) => {
                        for nm in list.nested {
                            if let NestedMeta::Meta(Meta::Path(path)) = nm {
                                if path.is_ident("ignore") {
                                    return quote::quote! {};
                                }
                            } else {
                                unimplemented!()
                            }
                        }
                    }
                    _ => unimplemented!(),
                }
            }
        }
        let field = f.ident.as_ref().map(|x| quote! {#x}).unwrap_or_else(|| {
            let i = syn::Index::from(i);
            quote! {#i}
        });
        let ty = &f.ty;
        let name = if f.ident.is_none() && n_fields == 1 {
            quote_spanned! {f.span()=> ""}
        } else {
            quote_spanned! {f.span()=> stringify!(#field)}
        };

        quote_spanned! {f.span()=>
            <#ty as crate::editor::Inspect>::#fn_name(&#maybe_mut self.#field, ui, #name);
        }
    });

    if n_fields == 0 {
        quote! {
            None
        }
    } else {
        quote! {
            Some(|| {
                #(#fields)*
            })
        }
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

    let generics = add_trait_bounds(di.generics.clone());
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics crate::editor::Inspect for #name #ty_generics #where_clause {
            fn inspect<'a>(&self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                crate::editor::inspect::inspect_struct(name, Some(stringify!(#name)),
                    ui, #body);
            }

            fn inspect_mut<'a>(&mut self, ui: &imgui::Ui<'a>, name: &str) {
                use crate::editor::Inspect;
                crate::editor::inspect::inspect_struct(name, Some(stringify!(#name)),
                    ui, #body_mut);
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
