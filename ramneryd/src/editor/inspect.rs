use imgui::im_str;

pub type Ui<'a> = crate::render::ui::UiFrame<'a>;

pub trait Inspect {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str);
    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str);
}

macro_rules! impl_inspect_direct {
    ($ty:ty, $imgui_ty:ident) => {
        impl Inspect for $ty {
            fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
                let mut v = *self;
                imgui::$imgui_ty::new(ui.inner(), &im_str!("{}", name), &mut v).build();
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
                imgui::$imgui_ty::new(ui.inner(), &im_str!("{}", name), self).build();
            }
        }
    };
}

impl_inspect_direct!(f32, InputFloat);
impl_inspect_direct!(i32, InputInt);

macro_rules! impl_inspect_cast {
    ($ty:ty, $imgui_ty:ident) => {
        impl Inspect for $ty {
            fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
                let mut v = *self as _;
                imgui::$imgui_ty::new(ui.inner(), &im_str!("{}", name), &mut v).build();
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
                let mut v = *self as _;
                imgui::$imgui_ty::new(ui.inner(), &im_str!("{}", name), &mut v).build();
                *self = v as _;
            }
        }
    };
}

impl_inspect_cast!(u32, InputInt);
impl_inspect_cast!(u16, InputInt);
impl_inspect_cast!(u8, InputInt);

macro_rules! impl_inspect_vec {
    ($ty:ident, $imgui_ty:ident, $n:expr) => {
        impl Inspect for crate::math::$ty {
            fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
                let mut v = self.into_array();
                imgui::$imgui_ty::new(ui.inner(), &im_str!("{}", name), &mut v)
                    .read_only(true)
                    .build();
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
                use std::convert::TryFrom;
                let v = <&mut [f32; $n]>::try_from(self.as_mut_slice()).unwrap();
                imgui::$imgui_ty::new(ui.inner(), &im_str!("{}", name), v).build();
            }
        }
    };
}

impl_inspect_vec!(Vec3, InputFloat3, 3);
impl_inspect_vec!(Rgb, InputFloat3, 3);
impl_inspect_vec!(Vec4, InputFloat4, 4);
impl_inspect_vec!(Rgba, InputFloat4, 4);

fn inspect_mat<'a>(m: &crate::math::Mat4, ui: &Ui<'a>, _name: &str) -> [[f32; 4]; 4] {
    let mut rows = m.into_row_arrays();
    for (i, mut row) in rows.iter_mut().enumerate() {
        imgui::InputFloat4::new(ui.inner(), &im_str!("{}", i), &mut row).build();
    }
    rows
}

impl Inspect for crate::math::Mat4 {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        let _ = inspect_mat(self, ui, name);
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        *self = Self::from_row_arrays(inspect_mat(self, ui, name));
    }
}

#[derive(Default, Debug)]
struct QuatEditState {
    axis: [f32; 3],
    angle_radians: f32,
}

impl Inspect for crate::math::Quat {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        let v = self.into_vec4();
        <crate::math::Vec4 as Inspect>::inspect(&v, ui, name);
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = self.into_vec4();
        <crate::math::Vec4 as Inspect>::inspect_mut(&mut v, ui, name);
        *self = Self::from_vec4(v);
        ui.inner().same_line(0.0);
        let id = format!("QuatEdit {}", name);
        let imgui_id = imgui::ImString::from(id.clone());
        if ui.inner().button(im_str!("edit"), [0.0, 0.0]) {
            ui.inner().open_popup(&imgui_id);
            ui.storage().insert(id.clone(), QuatEditState::default());
        }
        ui.inner().popup_modal(&imgui_id).build(|| {
            let mut storage = ui.storage();
            let state: &mut QuatEditState = storage
                .get_mut(&id)
                .expect("Got quat edit modal but no state resource");
            imgui::InputFloat3::new(ui.inner(), im_str!("axis"), &mut state.axis).build();
            imgui::InputFloat::new(
                ui.inner(),
                im_str!("angle (radians)"),
                &mut state.angle_radians,
            )
            .build();

            if ui.inner().button(im_str!("Apply"), [0.0; 2]) {
                *self = crate::math::Quat::rotation_3d(state.angle_radians, state.axis);
                ui.inner().close_current_popup();
            }
            ui.inner().same_line(0.0);
            if ui.inner().button(im_str!("Close"), [0.0; 2]) {
                ui.inner().close_current_popup();
            }
        });
    }
}

impl Inspect for crate::ecs::Entity {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(im_str!("{}: {:?}", name, self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

impl<T: Inspect> Inspect for Vec<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(im_str!("{}:", name));
        ui.inner().indent();
        for (i, e) in self.iter().enumerate() {
            let name = format!("[{}] ", i);
            e.inspect(ui, &name);
        }
        ui.inner().unindent();
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(im_str!("{}:", name));
        ui.inner().indent();
        for (i, e) in self.iter_mut().enumerate() {
            let name = format!("[{}] ", i);
            e.inspect_mut(ui, &name);
        }
        ui.inner().unindent();
    }
}

impl Inspect for String {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(im_str!("{}: {}", name, &self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = imgui::ImString::from(self.clone());
        if ui.inner().input_text(&im_str!("{}", name), &mut v).build() {
            *self = v.to_string();
        }
    }
}

impl Inspect for std::path::PathBuf {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(im_str!("{}: {}", name, &self.display()));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = imgui::ImString::from(
            self.clone()
                .into_os_string()
                .into_string()
                .unwrap_or_else(|_| String::from("bad path")),
        );
        if ui.inner().input_text(&im_str!("{}", name), &mut v).build() {
            *self = std::path::PathBuf::from(v.to_str());
        }
    }
}

impl<T> Inspect for resurs::Handle<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        let s = format!(
            "{}: Handle<{}>({})",
            name,
            std::any::type_name::<T>(),
            self.id()
        );
        ui.inner().text(&s);
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

impl<T> Inspect for trekanten::BufferHandle<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        let mut_str = if let trekanten::BufferMutability::Mutable = self.mutability() {
            "mut "
        } else {
            ""
        };

        let ty = std::any::type_name::<T>();
        let s = format!(
            "{}: &{}BufferHandle<{}>({})[{}..{}]",
            name,
            mut_str,
            ty,
            self.handle().id(),
            self.idx(),
            self.idx() + self.n_elems()
        );
        ui.inner().text(&s);
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

impl<T: Inspect> Inspect for Option<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        match self {
            None => ui.inner().text(format!("{}: None", name).as_str()),
            Some(x) => x.inspect(ui, name),
        };
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        match self {
            None => ui.inner().text(format!("{}: None", name).as_str()),
            Some(x) => x.inspect_mut(ui, name),
        };
    }
}

impl Inspect for bool {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(&imgui::im_str!("{}: {}", name, self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = *self;
        ui.inner().checkbox(&imgui::im_str!("{}", name), &mut v);
        *self = v;
    }
}

pub fn inspect_struct<'a, Body>(name: &str, ty: Option<&str>, ui: &Ui<'a>, body: Option<Body>)
where
    Body: FnMut(),
{
    if !name.is_empty() {
        ui.inner().text(name);
        ui.inner().same_line(0.0);
    }

    if let Some(ty) = ty {
        let id = if name.is_empty() {
            im_str!("struct {}", ty)
        } else {
            im_str!("struct {}##{}", ty, name)
        };
        let token = ui.inner().push_id(&id);
        if imgui::CollapsingHeader::new(&id)
            .leaf(body.is_none())
            .build(ui.inner())
        {
            ui.inner().indent();
            if let Some(mut body) = body {
                body();
            }
            ui.inner().unindent();
        }
        token.pop(ui.inner());
    } else {
        let token = ui.inner().push_id(&name);
        if let Some(mut body) = body {
            body();
        }
        token.pop(ui.inner());
    }
}

impl Inspect for trekanten::pipeline::PolygonMode {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        if !name.is_empty() {
            ui.inner().text(format!("{}:", name));
            ui.inner().same_line(0.0);
        }

        match self {
            Self::Fill => {
                let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), "Fill");
                let _ = imgui::CollapsingHeader::new(&ty)
                    .default_open(true)
                    .leaf(true)
                    .build(ui.inner());
            }
            Self::Line => {
                let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), "Line");
                let _ = imgui::CollapsingHeader::new(&ty)
                    .default_open(true)
                    .leaf(true)
                    .build(ui.inner());
            }
            Self::Point => {
                let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), "Point");
                let _ = imgui::CollapsingHeader::new(&ty)
                    .default_open(true)
                    .leaf(true)
                    .build(ui.inner());
            }
        }
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

// TODO: A generic impl_inspect_display here
impl Inspect for usize {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.inner().text(&im_str!("{}: {}", name, self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

impl<BT> Inspect for trekanten::mem::OwningBufferDescriptor<BT> {
    fn inspect<'a>(&self, ui: &Ui<'a>, _name: &str) {
        ui.inner().text("TODO")
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, _name: &str) {
        ui.inner().text("TODO")
    }
}

impl<T> Inspect for resurs::Async<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, _name: &str) {
        ui.inner().text("TODO")
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, _name: &str) {
        ui.inner().text("TODO")
    }
}

macro_rules! impl_todo_inspect {
    ($ty:path) => {
        impl Inspect for $ty {
            fn inspect<'a>(&self, ui: &Ui<'a>, _name: &str) {
                ui.inner().text("TODO");
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, _name: &str) {
                ui.inner().text("TODO");
            }
        }
    };
}

impl_todo_inspect!(trekanten::texture::TextureDescriptor);
impl_todo_inspect!(trekanten::texture::Texture);
