use crate::ecs::prelude::*;
use crate::visit::{Meta, MetaOrigin, Visitable, Visitor};

use std::borrow::Cow;
use std::collections::HashMap;

use crate::render::imgui::UiFrame;

pub type Ui<'a> = crate::render::imgui::UiFrame<'a>;

fn label<T>(m: &Meta<T>) -> Cow<str> {
    match &m.origin {
        MetaOrigin::NamedField { name } => Cow::Borrowed(*name),
        MetaOrigin::TupleField { idx } => Cow::Owned(format!("{}", idx)),
        MetaOrigin::Standalone => Cow::Owned(String::default()),
    }
}

fn push_id<'a, T>(frame: &'a UiFrame<'a>, m: &Meta<T>) -> imgui::IdStackToken<'a> {
    match &m.origin {
        MetaOrigin::NamedField { name } => frame.inner().push_id(*name),
        MetaOrigin::TupleField { idx } => {
            let idx: i32 = (*idx) as i32;
            frame.inner().push_id_int(idx)
        }
        MetaOrigin::Standalone => frame.inner().push_id_int(0),
    }
}

type InspectorCallback = for<'a> fn(&mut ImguiVisitor<'a>, &World, Entity);

#[derive(Default)]
pub struct Inspector {
    components: HashMap<std::any::TypeId, InspectorCallback>,
}

pub trait ImguiVisitableComponent<'a>: Visitable<ImguiVisitor<'a>> + Component + 'static {}
impl<'a, T: Visitable<ImguiVisitor<'a>> + Component + 'static> ImguiVisitableComponent<'a> for T {}

fn inspect_component<C>(v: &mut ImguiVisitor<'_>, world: &World, e: Entity)
where
    for<'b> C: ImguiVisitableComponent<'b>,
{
    let type_name = std::any::type_name::<C>();
    let size = std::mem::size_of::<C>();

    // Why do I need 3 function calls to get a &mut reference?! :(
    let mut storage = world.write_component::<C>();
    let Some(mut component) = storage.get_mut(e) else {
        return;
    };
    let component: &mut C = component.access_mut();

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
        self.components
            .insert(std::any::TypeId::of::<C>(), inspect_component::<C>);
    }

    pub fn inspect_components(&self, v: &mut ImguiVisitor<'_>, w: &World, e: Entity) {
        let visit = |info: &specs::ComponentInfo| {
            if let Some(callback) = self.components.get(&info.type_id) {
                callback(v, w, e);
            } else {
                let _open = imgui::CollapsingHeader::new(&imgui::ImString::from(String::from(
                    info.type_name,
                )))
                .leaf(true)
                .build(v.ui.inner());
            }
        };
        w.visit(e, visit).expect("Entity is not alive");
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
                self.ui.inner().text(format!("{}", idx));
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
                let token = self.ui.inner().push_id_int(id);
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
                let token = self.ui.inner().push_id_int(id);
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

macro_rules! impl_visit_scalar {
    ($ty:ty) => {
        impl<'a> Visitor<$ty> for ImguiVisitor<'a> {
            fn visit(&mut self, t: &$ty, m: &Meta<$ty>) {
                let mut copy: $ty = *t as _;
                self.ui
                    .inner()
                    .input_scalar(&label(m), &mut copy)
                    .read_only(true)
                    .build();
            }

            fn visit_mut(&mut self, t: &mut $ty, m: &Meta<$ty>) {
                let mut copy: $ty = *t;
                let changed = self.ui.inner().input_scalar(&label(m), &mut copy).build();
                if changed {
                    *t = copy;
                }
            }
        }
    };
}

impl_visit_scalar!(f64);
impl_visit_scalar!(f32);

impl_visit_scalar!(i64);
impl_visit_scalar!(i32);
impl_visit_scalar!(i16);
impl_visit_scalar!(i8);

impl_visit_scalar!(u64);
impl_visit_scalar!(u32);
impl_visit_scalar!(u16);
impl_visit_scalar!(u8);

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
use specs::storage::AccessMut;
use trekant::Texture;
impl_visit_todo_generic!(Async);
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
        self.ui.inner().modal_popup(&storage_id, || {
            let mut storage = self.ui.storage();
            let state: &mut QuatEditState = storage
                .get_mut(&storage_id)
                .expect("Got quat edit modal but no state resource");
            self.ui
                .inner()
                .input_float3("axis", &mut state.axis)
                .build();
            self.ui
                .inner()
                .input_float("angle (radians)", &mut state.angle_radians)
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

pub fn draw_struct_mut<'a, T: Visitable<ImguiVisitor<'a>>>(
    ui: &'a Ui<'a>,
    title: &'static str,
    v: &mut T,
) {
    let mut vis = ImguiVisitor::new(ui);
    vis.visit_mut(
        v,
        &Meta {
            type_name: title,
            range: None,
            origin: MetaOrigin::Standalone,
        },
    );
}
