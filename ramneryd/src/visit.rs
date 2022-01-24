use std::ops::Range;

/// Simple reflection utilites for custom types. Implement Reflect<V> for a struct/enum etc. and it can then
/// be traversed by any number of visitors (implementing Visit).

/// A struct carrying reflection information for a type
pub enum MetaOrigin {
    NamedField { name: &'static str },
    TupleField { idx: u8 },
    Standalone,
}

pub struct Meta<T> {
    pub type_name: &'static str,
    pub range: Option<Range<T>>,
    pub origin: MetaOrigin,
}

impl<T> Meta<T> {
    pub fn field(name: &'static str) -> Self {
        Meta {
            type_name: std::any::type_name::<T>(),
            range: None,
            origin: MetaOrigin::NamedField { name },
        }
    }
    pub fn standalone() -> Self {
        Meta {
            type_name: std::any::type_name::<T>(),
            range: None,
            origin: MetaOrigin::Standalone,
        }
    }
}

/// A trait that a visitor implements
pub trait Visitor<T> {
    fn visit(&mut self, t: &T, m: &Meta<T>);
    fn visit_mut(&mut self, t: &mut T, m: &Meta<T>);
}

// A trait that a type implements so that it can be introspected by a visitor
pub trait Visitable<V> {
    const IS_ENUM: bool;
    fn has_fields(&self) -> bool;
    fn visit_fields(&self, visitor: &mut V);
    fn visit_fields_mut(&mut self, visitor: &mut V);

    /// Only valid of IS_ENUM is true. Implementation is allowed to panic otherwise.
    fn variant_name(&self) -> &str;
    /// Only valid of IS_ENUM is true. Implementation is allowed to panic otherwise.
    fn variant_idx(&self) -> usize;
}

#[cfg(not)]
mod unused {
    struct PrintVisitor {
        indent: usize,
    }

    impl PrintVisitor {
        fn incr_indent(&mut self) {
            self.indent += 1;
        }

        fn decr_indent(&mut self) {
            self.indent -= 1;
        }

        fn indent(&self) -> String {
            (0..self.indent).map(|_| '\t').collect()
        }
    }

    impl Visitor<u32> for PrintVisitor {
        fn visit(&mut self, t: &u32, m: Meta<u32>) {
            println!(
                "{}{}: {}",
                self.indent(),
                m.field_name.unwrap_or(String::from("none")),
                t
            );
        }
    }

    impl Visitor<String> for PrintVisitor {
        fn visit(&mut self, t: &String, m: Meta<String>) {
            println!(
                "{}{}: {}",
                self.indent(),
                m.field_name.unwrap_or(String::from("none")),
                t
            );
        }
    }

    impl<T> Visitor<T> for PrintVisitor
    where
        T: Reflect<Self>,
    {
        fn visit(&mut self, t: &T, m: Meta<T>) {
            let field_name = m
                .field_name
                .map(|f| format!("{}: ", f))
                .unwrap_or(String::default());
            println!("{}{}{} {{", self.indent(), field_name, m.type_name,);
            self.incr_indent();
            t.reflect(self);
            self.decr_indent();
            println!("{}}}", self.indent());
        }
    }

    struct A {
        x: u32,
        y: String,
    }

    impl<V> Reflect<V> for A
    where
        V: Visitor<u32> + Visitor<String>,
    {
        fn reflect(&self, v: &mut V) {
            let m = Meta {
                type_name: String::from("u32"),
                field_name: Some(String::from("x")),
                range: Some(Range { start: 0, end: 10 }),
            };
            let m2 = Meta {
                type_name: String::from("String"),
                field_name: Some(String::from("y")),
                range: None,
            };
            v.visit(&self.x, m);
            v.visit(&self.y, m2);
        }
    }

    struct B {
        a: A,
        b: u32,
    }

    impl<V> Reflect<V> for B
    where
        V: Visitor<u32> + Visitor<A>,
    {
        fn reflect(&self, v: &mut V) {
            let m = Meta {
                type_name: String::from("A"),
                field_name: Some(String::from("a")),
                range: None,
            };
            let m2 = Meta {
                type_name: String::from("u32"),
                field_name: Some(String::from("b")),
                range: Some(Range { start: 5, end: 10 }),
            };
            v.visit(&self.a, m);
            v.visit(&self.b, m2);
        }
    }

    struct C {
        a: A,
        b: B,
    }

    impl<V> Reflect<V> for C
    where
        V: Visitor<A> + Visitor<B>,
    {
        fn reflect(&self, v: &mut V) {
            let m = Meta {
                type_name: String::from("A"),
                field_name: Some(String::from("a")),
                range: None,
            };
            let m2 = Meta {
                type_name: String::from("B"),
                field_name: Some(String::from("b")),
                range: None,
            };
            v.visit(&self.a, m);
            v.visit(&self.b, m2);
        }
    }
}
