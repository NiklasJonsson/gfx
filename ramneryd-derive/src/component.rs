use proc_macro2::TokenStream;
use quote::quote;
use syn::{spanned::Spanned, DeriveInput, Lit, Meta, NestedMeta, Path};

pub(crate) fn impl_component(di: &DeriveInput) -> TokenStream {
    let name = &di.ident;
    let name_caps = quote::format_ident!(
        "_RAMNERYD_META_COMPONENT_{}",
        &name.to_string().to_uppercase()
    );
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
                            NestedMeta::Meta(Meta::Path(path)) => {
                                if path.is_ident("visitable") {
                                    panic!("Unsupported visitable");
                                }
                                if path.is_ident("inspect") {
                                    panic!("Deprecated inspect");
                                }
                            }
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

    let meta_component = quote::quote! {
        crate::ecs::meta::Component {
            name: stringify!(#name),
            size: std::mem::size_of::<#name>(),
            has: <#name>::has,
            register: <#name>::register,
        }
    };

    // TODO: meta() can be const when we have function pointer as const
    quote! {
        /// specs
        impl #impl_generics crate::ecs::prelude::Component for #name #ty_generics #where_clause {
            type Storage = crate::ecs::prelude::#storage<Self>;
        }

        /// ECS-related
        impl #impl_generics #name #ty_generics #where_clause {
            pub fn has(world: &crate::ecs::World, ent: crate::ecs::Entity) -> bool {
                use crate::ecs::prelude::WorldExt;
                world.read_storage::<Self>().get(ent).is_some()
            }

            pub fn register(world: &mut crate::ecs::World) {
                use crate::ecs::prelude::WorldExt;
                world.register::<Self>();
            }
        }

        /// Meta function
        impl #impl_generics #name #ty_generics #where_clause {
            // TODO: Make const when fn pointers are stable
            pub fn meta() -> crate::ecs::meta::Component {
                #meta_component
            }
        }

        // TODO: Use meta() here when const
        #[linkme::distributed_slice(crate::ecs::meta::ALL_COMPONENTS)]
        static #name_caps: crate::ecs::meta::Component = #meta_component;
    }
}
