use crate::ecs::prelude::*;
use crate::visit::{Meta, MetaOrigin, Visitable, Visitor};

use std::borrow::Cow;

use crate::render::ui::UiFrame;

pub type Ui<'a> = crate::render::ui::UiFrame<'a>;

fn label<T>(m: &Meta<T>) -> Cow<str> {
    match &m.origin {
        MetaOrigin::NamedField { name } => Cow::Borrowed(*name),
        MetaOrigin::TupleField { idx } => Cow::Owned(format!("{}", idx)),
        MetaOrigin::Standalone => Cow::Owned(String::default()),
    }
}

fn push_id<'a, T>(frame: &UiFrame<'a>, m: &Meta<T>) -> imgui::IdStackToken<'a> {
    match &m.origin {
        MetaOrigin::NamedField { name } => frame.inner().push_id(*name),
        MetaOrigin::TupleField { idx } => {
            let idx: i32 = (*idx)
                .try_into()
                .expect("Tuple idx should always fit into i32");
            frame.inner().push_id(idx)
        }
        MetaOrigin::Standalone => frame.inner().push_id(0),
    }
}

type InspectorCallback = for<'a> fn(&mut ImguiVisitor<'a>, &World, Entity);

#[derive(Default)]
pub struct Inspector {
    components: Vec<InspectorCallback>,
}

pub trait ImguiVisitableComponent<'a>: Visitable<ImguiVisitor<'a>> + Component + 'static {}
impl<'a, T: Visitable<ImguiVisitor<'a>> + Component + 'static> ImguiVisitableComponent<'a> for T {}

fn inspect_component<'a, C>(v: &mut ImguiVisitor<'a>, world: &World, e: Entity)
where
    for<'b> C: ImguiVisitableComponent<'b>,
{
    let type_name = std::any::type_name::<C>();
    let size = std::mem::size_of::<C>();
    let mut storage = world.write_component::<C>();

    let component = if let Some(component) = storage.get_mut(e) {
        component
    } else {
        return;
    };

    if size == 0 {
        let _open = imgui::CollapsingHeader::new(&imgui::ImString::from(String::from(type_name)))
            .leaf(true)
            .build(v.ui.inner());
    } else {
        v.visit_mut(
            component,
            &Meta {
                range: None,
                type_name,
                origin: MetaOrigin::Standalone,
            },
        );
    }
}
impl Inspector {
    pub fn add<C>(&mut self)
    where
        for<'a> C: ImguiVisitableComponent<'a>,
    {
        self.components.push(inspect_component::<C>);
    }

    pub fn inspect_components<'a>(&self, v: &mut ImguiVisitor<'a>, w: &World, e: Entity) {
        for comp in &self.components {
            comp(v, w, e);
        }
    }
}

pub struct ImguiVisitor<'a> {
    ui: &'a Ui<'a>,
}

impl<'a> ImguiVisitor<'a> {
    pub fn new(ui: &'a Ui) -> Self {
        Self { ui }
    }
}

impl<'a> ImguiVisitor<'a> {
    fn mk_field<T>(&self, m: &Meta<T>) {
        match &m.origin {
            MetaOrigin::NamedField { name } => {
                self.ui.inner().text(name);
                self.ui.inner().same_line();
            }
            MetaOrigin::TupleField { idx } => {
                self.ui.inner().text(&format!("{}", idx));
                self.ui.inner().same_line();
            }
            MetaOrigin::Standalone => (),
        }
    }
}

impl<'a> ImguiVisitor<'a> {
    fn visit_visitable_begin<T: Visitable<Self>>(
        &self,
        t: &T,
        m: &Meta<T>,
    ) -> Option<imgui::IdStackToken<'a>> {
        self.mk_field(m);

        let token = push_id(self.ui, m);

        let header_label = if T::IS_ENUM {
            format!("enum {}::{}", m.type_name, t.variant_name())
        } else {
            format!("struct {}", m.type_name)
        };
        if m.type_name.is_empty() {
            // We don't want a header for this if there is a type name. Still, add a newline for the comming fields
            self.ui.inner().new_line();
            self.ui.inner().indent();
            Some(token)
        } else if imgui::CollapsingHeader::new(&header_label)
            .leaf(!t.has_fields())
            .build(self.ui.inner())
        {
            self.ui.inner().indent();
            Some(token)
        } else {
            token.pop();
            None
        }
    }

    fn visit_visitable_end(&self, token: imgui::IdStackToken) {
        self.ui.inner().unindent();
        token.pop();
    }
}

impl<'a, T> Visitor<T> for ImguiVisitor<'a>
where
    T: Visitable<Self>,
{
    fn visit(&mut self, t: &T, m: &Meta<T>) {
        if let Some(outer_token) = self.visit_visitable_begin(t, m) {
            t.visit_fields(self);
            self.visit_visitable_end(outer_token);
        }
    }

    fn visit_mut(&mut self, t: &mut T, m: &Meta<T>) {
        if let Some(outer_token) = self.visit_visitable_begin(t, m) {
            t.visit_fields_mut(self);
            self.visit_visitable_end(outer_token);
        }
    }
}

impl<'a> ImguiVisitor<'a> {
    fn visit_array_begin<T>(&mut self, m: &Meta<T>) -> Option<imgui::IdStackToken<'a>> {
        self.mk_field(m);
        let header_label = m.type_name;
        let token = push_id(self.ui, m);

        if imgui::CollapsingHeader::new(&header_label).build(self.ui.inner()) {
            self.ui.inner().indent();
            Some(token)
        } else {
            token.pop();
            None
        }
    }

    fn visit_array_end(&self, token: imgui::IdStackToken<'a>) {
        self.ui.inner().unindent();
        token.pop();
    }
}

impl<'a, T, const N: usize> Visitor<[T; N]> for ImguiVisitor<'a>
where
    T: Visitable<Self>,
{
    fn visit(&mut self, t: &[T; N], m: &Meta<[T; N]>) {
        if let Some(outer_token) = self.visit_array_begin(m) {
            for (i, e) in t.iter().enumerate() {
                let id: i32 = i.try_into().unwrap();
                let token = self.ui.inner().push_id(id);
                let idx: u8 = i.try_into().expect("Too many elements");
                self.visit(
                    e,
                    &Meta {
                        type_name: std::any::type_name::<T>(),
                        range: None,
                        origin: MetaOrigin::TupleField { idx },
                    },
                );
                token.pop();
            }
            self.visit_array_end(outer_token);
        }
    }

    fn visit_mut(&mut self, t: &mut [T; N], m: &Meta<[T; N]>) {
        if let Some(outer_token) = self.visit_array_begin(m) {
            for (i, e) in t.iter_mut().enumerate() {
                let id: i32 = i.try_into().unwrap();
                let token = self.ui.inner().push_id(id);
                let idx: u8 = i.try_into().expect("Too many elements");
                self.visit_mut(
                    e,
                    &Meta {
                        type_name: std::any::type_name::<T>(),
                        range: None,
                        origin: MetaOrigin::TupleField { idx },
                    },
                );
                token.pop();
            }
            self.visit_array_end(outer_token);
        }
    }
}

macro_rules! impl_visit_cast {
    ($ty:ty, $imgui_ty:ident) => {
        impl<'a> Visitor<$ty> for ImguiVisitor<'a> {
            fn visit(&mut self, t: &$ty, m: &Meta<$ty>) {
                let mut v = *t as _;
                imgui::$imgui_ty::new(self.ui.inner(), &label(m), &mut v)
                    .read_only(true)
                    .build();
            }

            fn visit_mut(&mut self, t: &mut $ty, m: &Meta<$ty>) {
                let mut v = *t as _;
                imgui::$imgui_ty::new(self.ui.inner(), &label(m), &mut v).build();
                *t = v as _;
            }
        }
    };
}

impl_visit_cast!(f32, InputFloat);
impl_visit_cast!(i32, InputInt);
impl_visit_cast!(u32, InputInt);
impl_visit_cast!(u16, InputInt);
impl_visit_cast!(u8, InputInt);

macro_rules! impl_visit_array {
    ($ty:ty, $imgui_ty:ident) => {
        impl<'a> Visitor<$ty> for ImguiVisitor<'a> {
            fn visit(&mut self, t: &$ty, m: &Meta<$ty>) {
                let mut copy = *t;
                imgui::$imgui_ty::new(self.ui.inner(), &label(m), &mut copy)
                    .read_only(true)
                    .build();
            }

            fn visit_mut(&mut self, t: &mut $ty, m: &Meta<$ty>) {
                imgui::$imgui_ty::new(self.ui.inner(), &label(m), t).build();
            }
        }
    };
}

impl_visit_array!([f32; 3], InputFloat3);

macro_rules! impl_visit_into_array {
    ($ty:ident, $imgui_ty:ident, $n:expr) => {
        impl<'a> Visitor<$ty> for ImguiVisitor<'a> {
            fn visit(&mut self, t: &$ty, m: &Meta<$ty>) {
                let mut v = t.into_array();
                imgui::$imgui_ty::new(self.ui.inner(), &label(m), &mut v)
                    .read_only(true)
                    .build();
            }

            fn visit_mut(&mut self, t: &mut $ty, m: &Meta<$ty>) {
                use std::convert::TryFrom;
                let v = <&mut [f32; $n]>::try_from(t.as_mut_slice()).unwrap();
                imgui::$imgui_ty::new(self.ui.inner(), &label(m), v).build();
            }
        }
    };
}

use crate::math::{Rgb, Rgba, Vec3, Vec4};
impl_visit_into_array!(Vec3, InputFloat3, 3);
impl_visit_into_array!(Rgb, InputFloat3, 3);
impl_visit_into_array!(Vec4, InputFloat4, 4);
impl_visit_into_array!(Rgba, InputFloat4, 4);

macro_rules! impl_visit_display {
    ($ty:ident) => {
        impl<'a> Visitor<$ty> for ImguiVisitor<'a> {
            fn visit(&mut self, t: &$ty, m: &Meta<$ty>) {
                self.ui.inner().text(format!("{}: {}", &label(m), t));
            }
            fn visit_mut(&mut self, t: &mut $ty, m: &Meta<$ty>) {
                self.visit(t, m);
            }
        }
    };
}

impl_visit_display!(usize);

macro_rules! impl_visit_todo {
    ($ty:ident) => {
        impl<'a> Visitor<$ty> for ImguiVisitor<'a> {
            #[allow(unused)]
            fn visit(&mut self, t: &$ty, m: &Meta<$ty>) {
                self.ui.inner().text("TODO");
            }

            fn visit_mut(&mut self, t: &mut $ty, m: &Meta<$ty>) {
                self.visit(t, m);
            }
        }
    };
}

macro_rules! impl_visit_todo_generic {
    ($ty:ident) => {
        #[allow(dead_code)]
        impl<'a, T> Visitor<$ty<T>> for ImguiVisitor<'a> {
            #[allow(unused)]
            fn visit(&mut self, t: &$ty<T>, m: &Meta<$ty<T>>) {
                self.ui.inner().text("TODO");
            }

            fn visit_mut(&mut self, t: &mut $ty<T>, m: &Meta<$ty<T>>) {
                self.visit(t, m);
            }
        }
    };
}

use resurs::Async;
use trekanten::texture::{Texture, TextureDescriptor};
impl_visit_todo_generic!(Async);
impl_visit_todo!(TextureDescriptor);
impl_visit_todo!(Texture);

impl<'a> Visitor<bool> for ImguiVisitor<'a> {
    fn visit(&mut self, t: &bool, m: &Meta<bool>) {
        let mut tmp = *t;
        self.ui.inner().checkbox(&label(m), &mut tmp);
    }

    fn visit_mut(&mut self, t: &mut bool, m: &Meta<bool>) {
        self.ui.inner().checkbox(&label(m), t);
    }
}

use crate::ecs::Entity;
impl<'a> Visitor<Entity> for ImguiVisitor<'a> {
    fn visit(&mut self, t: &Entity, m: &Meta<Entity>) {
        self.ui.inner().text(format!("{}: {}", &label(m), t.id()));
    }

    fn visit_mut(&mut self, t: &mut Entity, m: &Meta<Entity>) {
        // Read only for entities
        self.visit(t, m);
    }
}

#[derive(Default, Debug)]
struct QuatEditState {
    axis: [f32; 3],
    angle_radians: f32,
}

impl<'a> Visitor<crate::math::Quat> for ImguiVisitor<'a> {
    fn visit(&mut self, t: &crate::math::Quat, m: &Meta<crate::math::Quat>) {
        let mut v = t.into_vec4().into_array();
        imgui::InputFloat4::new(self.ui.inner(), &label(m), &mut v)
            .read_only(true)
            .build();
    }

    fn visit_mut(&mut self, t: &mut crate::math::Quat, m: &Meta<crate::math::Quat>) {
        let mut v = t.into_vec4().into_array();
        imgui::InputFloat4::new(self.ui.inner(), &label(m), &mut v).build();
        *t = crate::math::Quat::from(crate::math::Vec4::from(v));

        self.ui.inner().same_line();
        let storage_id = String::from("Edit quaternion");
        if self.ui.inner().button("edit") {
            self.ui.inner().open_popup(&storage_id);
            self.ui
                .storage()
                .insert(storage_id.clone(), QuatEditState::default());
        }
        self.ui
            .inner()
            .popup_modal(&storage_id)
            .build(self.ui.inner(), || {
                let mut storage = self.ui.storage();
                let state: &mut QuatEditState = storage
                    .get_mut(&storage_id)
                    .expect("Got quat edit modal but no state resource");
                imgui::InputFloat3::new(self.ui.inner(), "axis", &mut state.axis).build();
                imgui::InputFloat::new(
                    self.ui.inner(),
                    "angle (radians)",
                    &mut state.angle_radians,
                )
                .build();

                if self.ui.inner().button("Apply") {
                    *t = crate::math::Quat::rotation_3d(state.angle_radians, state.axis);
                    self.ui.inner().close_current_popup();
                }
                self.ui.inner().same_line();
                if self.ui.inner().button("Close") {
                    self.ui.inner().close_current_popup();
                }
            });
    }
}

impl<'a> Visitor<String> for ImguiVisitor<'a> {
    fn visit(&mut self, t: &String, m: &Meta<String>) {
        self.ui.inner().text(format!("{}: {}", label(m), &t));
    }

    fn visit_mut(&mut self, t: &mut String, m: &Meta<String>) {
        self.ui.inner().text(format!("{}: {}", label(m), &t));
        if imgui::InputText::new(self.ui.inner(), &label(m), t).build() {}
    }
}
