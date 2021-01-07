use imgui::{im_str, Ui};

pub trait Inspect {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str);
    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str);
}

macro_rules! impl_inspect_direct {
    ($ty:ty, $imgui_ty:ident) => {
        impl Inspect for $ty {
            fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
                let mut v = *self;
                imgui::$imgui_ty::new(ui, &im_str!("{}", name), &mut v).build();
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
                imgui::$imgui_ty::new(ui, &im_str!("{}", name), self).build();
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
                imgui::$imgui_ty::new(ui, &im_str!("{}", name), &mut v).build();
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
                let mut v = *self as _;
                imgui::$imgui_ty::new(ui, &im_str!("{}", name), &mut v).build();
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
                imgui::$imgui_ty::new(ui, &im_str!("{}", name), &mut v)
                    .read_only(true)
                    .build();
            }

            fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
                use std::convert::TryFrom;
                let v = <&mut [f32; $n]>::try_from(self.as_mut_slice()).unwrap();
                imgui::$imgui_ty::new(ui, &im_str!("{}", name), v).build();
            }
        }
    };
}

impl_inspect_vec!(Vec3, InputFloat3, 3);
impl_inspect_vec!(Vec4, InputFloat4, 4);

fn inspect_mat<'a>(m: &crate::math::Mat4, ui: &Ui<'a>, _name: &str) -> [[f32; 4]; 4] {
    let mut rows = m.into_row_arrays();
    for (i, mut row) in rows.iter_mut().enumerate() {
        imgui::InputFloat4::new(ui, &im_str!("{}", i), &mut row).build();
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

impl Inspect for crate::math::Quat {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        let v = self.into_vec4();
        <crate::math::Vec4 as Inspect>::inspect(&v, ui, name);
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = self.into_vec4();
        <crate::math::Vec4 as Inspect>::inspect_mut(&mut v, ui, name);
        *self = Self::from_vec4(v);
    }
}

impl Inspect for crate::ecs::Entity {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.text(im_str!("{}: {:?}", name, self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

impl<T: Inspect> Inspect for Vec<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.text(im_str!("{}:", name));
        ui.indent();
        for (i, e) in self.iter().enumerate() {
            let name = format!("[{}] ", i);
            e.inspect(ui, &name);
        }
        ui.unindent();
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        ui.text(im_str!("{}:", name));
        ui.indent();
        for (i, e) in self.iter_mut().enumerate() {
            let name = format!("[{}] ", i);
            e.inspect_mut(ui, &name);
        }
        ui.unindent();
    }
}

impl Inspect for String {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.text(im_str!("{}: {}", name, &self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = imgui::ImString::from(self.clone());
        if ui.input_text(&im_str!("{}", name), &mut v).build() {
            *self = v.to_string();
        }
    }
}

impl Inspect for std::path::PathBuf {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.text(im_str!("{}: {}", name, &self.display()));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = imgui::ImString::from(
            self.clone()
                .into_os_string()
                .into_string()
                .unwrap_or_else(|_| String::from("bad path")),
        );
        if ui.input_text(&im_str!("{}", name), &mut v).build() {
            *self = std::path::PathBuf::from(v.to_str());
        }
    }
}

impl<T> Inspect for resurs::Handle<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        let s = format!("{}: {}({})", name, std::any::type_name::<Self>(), self.id());
        ui.text(&s);
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
            "{}: &{}{}({})[{}..{}]",
            name,
            mut_str,
            ty,
            self.handle().id(),
            self.idx(),
            self.idx() + self.n_elems()
        );
        ui.text(&s);
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}

impl<T: Inspect> Inspect for Option<T> {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        match self {
            None => ui.text(format!("{}: None", name).as_str()),
            Some(x) => x.inspect(ui, name),
        };
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        match self {
            None => ui.text(format!("{}: None", name).as_str()),
            Some(x) => x.inspect_mut(ui, name),
        };
    }
}

impl Inspect for bool {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        ui.text(&imgui::im_str!("{}: {}", name, self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        let mut v = *self;
        ui.checkbox(&imgui::im_str!("{}", name), &mut v);
        *self = v;
    }
}

pub fn inspect_struct<'a, Body>(name: &str, ty: Option<&str>, ui: &Ui<'a>, body: Option<Body>)
where
    Body: FnMut(),
{
    if !name.is_empty() {
        ui.text(name);
        ui.same_line(0.0);
    }

    if let Some(ty) = ty {
        if imgui::CollapsingHeader::new(&im_str!("struct {}##{}", ty, name))
            .leaf(body.is_none())
            .build(ui)
        {
            ui.indent();
            if let Some(mut body) = body {
                body();
            }
            ui.unindent();
        }
    } else {
        if let Some(mut body) = body {
            body();
        }
    }
}

impl Inspect for trekanten::mesh::Mesh {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        inspect_struct(
            name,
            Some(std::any::type_name::<Self>()),
            ui,
            Some(|| {
                self.vertex_buffer.inspect(ui, "vertex_buffer");
                self.index_buffer.inspect(ui, "index_buffer");
            }),
        );
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        inspect_struct(
            name,
            Some(std::any::type_name::<Self>()),
            ui,
            Some(|| {
                self.vertex_buffer.inspect_mut(ui, "vertex_buffer");
                self.index_buffer.inspect_mut(ui, "index_buffer");
            }),
        );
    }
}

impl Inspect for trekanten::pipeline::PolygonMode {
    fn inspect<'a>(&self, ui: &Ui<'a>, name: &str) {
        if !name.is_empty() {
            ui.text(format!("{}:", name));
            ui.same_line(0.0);
        }

        match self {
            Self::Fill => {
                let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), "Fill");
                let _ = imgui::CollapsingHeader::new(&ty)
                    .default_open(true)
                    .leaf(true)
                    .build(ui);
            }
            Self::Line => {
                let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), "Line");
                let _ = imgui::CollapsingHeader::new(&ty)
                    .default_open(true)
                    .leaf(true)
                    .build(ui);
            }
            Self::Point => {
                let ty = imgui::im_str!("enum {}::{}", std::any::type_name::<Self>(), "Point");
                let _ = imgui::CollapsingHeader::new(&ty)
                    .default_open(true)
                    .leaf(true)
                    .build(ui);
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
        ui.text(&imgui::im_str!("{}: {}", name, self));
    }

    fn inspect_mut<'a>(&mut self, ui: &Ui<'a>, name: &str) {
        self.inspect(ui, name);
    }
}
