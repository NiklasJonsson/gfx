use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote, quote_spanned};
use syn::{
    parse_quote, spanned::Spanned as _, DeriveInput, Fields, GenericParam, Generics, Meta,
    NestedMeta,
};

fn visit_fn_name(is_mut: bool) -> syn::Ident {
    if is_mut {
        Ident::new("visit_mut", Span::call_site())
    } else {
        Ident::new("visit", Span::call_site())
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

/// Adds a bound for Visitable on all generic parameters in Generics
fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(ref mut type_param) = *param {
            type_param
                .bounds
                .push(parse_quote!(crate::visit::Visitable<V>));
        }
    }
    generics
}

#[derive(Debug)]
enum AttributeError {
    Parse { msg: String, span: Span },
    Unrecognized { span: Span },
}

fn visit_should_ignore_field(f: &syn::Field) -> Result<bool, AttributeError> {
    for attr in f.attrs.iter() {
        let span = attr.span();
        if attr.path.is_ident("visitable") {
            match attr.parse_meta() {
                Err(e) => {
                    let msg = format!("Failed to parse component attributes: {}", e);
                    return Err(AttributeError::Parse { msg, span });
                }
                Ok(Meta::List(list)) => {
                    for nm in list.nested {
                        if let NestedMeta::Meta(Meta::Path(path)) = nm {
                            if path.is_ident("ignore") {
                                return Ok(true);
                            }
                        } else {
                            return Err(AttributeError::Unrecognized { span });
                        }
                    }
                }
                _ => {
                    return Err(AttributeError::Parse {
                        msg: String::from("visitable attributes have to be a list"),
                        span,
                    })
                }
            }
        }
    }

    Ok(false)
}

fn add_visit_bounds(mut generics: Generics, data: &syn::Data) -> Generics {
    let mut visitor_pred = syn::PredicateType {
        lifetimes: None,
        bounded_ty: parse_quote!(V),
        colon_token: parse_quote!(:),
        bounds: Default::default(),
    };

    let push_bounds = |visitor_pred: &mut syn::PredicateType, fields: &Fields| {
        for f in fields.iter() {
            if visit_should_ignore_field(f).expect("Failed to read attributes") {
                continue;
            }
            let ty = &f.ty;
            visitor_pred
                .bounds
                .push(parse_quote!(crate::visit::Visitor<#ty>));
        }
    };

    match data {
        syn::Data::Struct(data) => {
            push_bounds(&mut visitor_pred, &data.fields);
        }
        syn::Data::Enum(data) => {
            for v in data.variants.iter() {
                push_bounds(&mut visitor_pred, &v.fields);
            }
        }
        syn::Data::Union(_) => unimplemented!("add_visit_bounds not implemented for Union"),
    }
    generics
        .make_where_clause()
        .predicates
        .push(syn::WherePredicate::Type(visitor_pred));

    generics
}

struct VisitGenerics {
    type_generics: Generics,
    impl_generics: Generics,
    where_generics: Generics,
}

impl VisitGenerics {
    fn new(di: &DeriveInput) -> Self {
        let type_generics = add_trait_bounds(di.generics.clone());

        let impl_generics = {
            let mut g = type_generics.clone();
            g.params.push(parse_quote!(V));
            g
        };

        let where_generics = {
            let g = type_generics.clone();
            add_visit_bounds(g, &di.data)
        };

        Self {
            type_generics,
            impl_generics,
            where_generics,
        }
    }

    fn split(
        &self,
    ) -> (
        syn::ImplGenerics,
        syn::TypeGenerics,
        Option<&syn::WhereClause>,
    ) {
        let (_, type_generics, _) = self.type_generics.split_for_impl();
        let (impl_generics, _, _) = self.impl_generics.split_for_impl();
        let (_, _, where_clause) = self.where_generics.split_for_impl();

        (impl_generics, type_generics, where_clause)
    }
}
fn visitable_enum_body(data: &syn::DataEnum, is_mut: bool) -> TokenStream {
    let fn_name = visit_fn_name(is_mut);
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
            let field_accessor = match &f.ident {
                Some(i) => quote_spanned! {f.ident.span()=>#i},
                None => {
                    let i = format_ident!("field_{}", i);
                    quote_spanned! {f.span()=> #i}
                }
            };

            let field_origin = match &f.ident {
                Some(i) => quote_spanned! { f.ident.span()=> crate::visit::MetaOrigin::NamedField { name: stringify!(#i) } },
                None => {
                    let i = syn::LitInt::new(format!("{}", i).as_str(), f.ident.span());
                    quote_spanned! {f.span()=> crate::visit::MetaOrigin::TupleField { idx: #i }}
                }
            };

            quote_spanned! {f.span()=>
                v.#fn_name(#field_accessor, &crate::visit::Meta {
                    type_name: stringify!(#ty),
                    range: None,
                    origin: #field_origin,
                });
            }
        });

        let fields_rhs = quote! {
            #(#fields_rhs)*
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MatchReturnType {
    Name,
    Index,
    HasFields,
}

fn visitable_enum_match(data: &syn::DataEnum, ty: MatchReturnType) -> TokenStream {
    let variants_impl = data.variants.iter().enumerate().map(|(i, variant)| {
        let id = &variant.ident;
        let fields_lhs = match variant.fields {
            Fields::Named(_) => quote_spanned! {variant.span() => { .. }},
            Fields::Unnamed(_) => quote_spanned! {variant.span() => ( .. )},
            Fields::Unit => quote! {},
        };

        if ty == MatchReturnType::Name {
            quote_spanned! {variant.span()=>
                Self::#id #fields_lhs => { stringify!(#id) }
            }
        } else if ty == MatchReturnType::HasFields {
            let has_fields = if variant.fields.is_empty() {
                quote! { false }
            } else {
                quote! { true }
            };
            quote_spanned! {variant.span()=>
                Self::#id #fields_lhs => { #has_fields }
            }
        } else {
            assert_eq!(ty, MatchReturnType::Index);
            quote_spanned! {variant.span()=>
                Self::#id #fields_lhs => { #i }
            }
        }
    });

    quote! {
        match self {
            #(#variants_impl)*
        }
    }
}

fn visitable_enum(di: &DeriveInput) -> TokenStream {
    let name = &di.ident;

    let data = match &di.data {
        syn::Data::Enum(data) => data,
        _ => panic!("Bad arg, expected enum"),
    };
    let visitable_body = visitable_enum_body(data, false);
    let visitable_body_mut = visitable_enum_body(data, true);

    let variant_idx_body = visitable_enum_match(data, MatchReturnType::Index);
    let variant_name_body = visitable_enum_match(data, MatchReturnType::Name);
    let has_fields_body = visitable_enum_match(data, MatchReturnType::HasFields);

    let generics = VisitGenerics::new(di);
    let (impl_generics, ty_generics, where_clause) = generics.split();
    quote! {
        impl #impl_generics crate::visit::Visitable<V> for #name #ty_generics #where_clause {
            const IS_ENUM: bool = true;

            fn has_fields(&self) -> bool {
                #has_fields_body
            }

            fn visit_fields(&self, v: &mut V) {
                #visitable_body
            }

            fn visit_fields_mut(&mut self, v: &mut V) {
                #visitable_body_mut
            }

            fn variant_idx(&self) -> usize {
                #variant_idx_body
            }

            fn variant_name(&self) -> &str {
                #variant_name_body
            }
        }
    }
}

fn visitable_struct_data(data: &syn::DataStruct, is_mut: bool) -> TokenStream {
    let fn_name = visit_fn_name(is_mut);
    let maybe_mut = maybe_mut(is_mut);
    let n_fields = data.fields.len();

    let fields = data.fields.iter().enumerate().map(|(i, f)| {
        match visit_should_ignore_field(f) {
            Ok(true) => return quote::quote! {},
            Ok(false) => (),
            Err(AttributeError::Parse { msg, span }) => {
                let msg = &msg;
                return quote::quote_spanned! {span=>
                    compile_error!(#msg);
                };
            }
            Err(AttributeError::Unrecognized { span }) => {
                let msg = "Unrecognized attribute";
                return quote::quote_spanned! {span=>
                    compile_error!(#msg);
                };
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
            {
                v.#fn_name(&#maybe_mut self.#field, &crate::visit::Meta {
                    type_name: stringify!(#ty),
                    range: None,
                    origin: crate::visit::MetaOrigin::NamedField {
                        name: #name,
                    }
                });
            }
        }
    });

    if n_fields == 0 {
        quote! {}
    } else {
        quote! {
            #(#fields)*
        }
    }
}

fn visitable_struct(di: &DeriveInput) -> TokenStream {
    let name = &di.ident;

    let data = match &di.data {
        syn::Data::Struct(data) => data,
        _ => panic!("Internal error: should be struct"),
    };

    let visitable_body = visitable_struct_data(data, false);
    let visitable_body_mut = visitable_struct_data(data, true);

    let has_fields_body = if data.fields.is_empty() {
        quote! { false }
    } else {
        quote! { true }
    };

    let generics = VisitGenerics::new(di);
    let (impl_generics, ty_generics, where_clause) = generics.split();

    quote! {
        impl #impl_generics crate::visit::Visitable<V> for #name #ty_generics #where_clause {
            const IS_ENUM: bool = false;
            fn has_fields(&self) -> bool {
                #has_fields_body
            }

            fn visit_fields(&self, v: &mut V) {
                #visitable_body
            }

            fn visit_fields_mut(&mut self, v: &mut V) {
                #visitable_body_mut
            }

            fn variant_idx(&self) -> usize {
                panic!("Not an enum");
            }

            fn variant_name(&self) -> &str {
                panic!("Not an enum");
            }
        }
    }
}
pub(crate) fn impl_visitable(di: &DeriveInput) -> TokenStream {
    let visitable_impl = match &di.data {
        syn::Data::Struct(_) => visitable_struct(di),
        syn::Data::Enum(_) => visitable_enum(di),
        _ => unimplemented!("Only Struct or enum is supported"),
    };

    quote! {
        #visitable_impl
    }
}
