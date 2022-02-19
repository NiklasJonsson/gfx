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

mod impl_std {
    use super::{Meta, Visitable, Visitor};

    impl<T, V> Visitable<V> for std::ops::Range<T>
    where
        V: Visitor<T>,
    {
        const IS_ENUM: bool = false;
        fn has_fields(&self) -> bool {
            true
        }

        fn visit_fields(&self, visitor: &mut V) {
            visitor.visit(&self.start, &Meta::field("start"));
            visitor.visit(&self.end, &Meta::field("end"));
        }

        fn visit_fields_mut(&mut self, visitor: &mut V) {
            visitor.visit_mut(&mut self.start, &Meta::field("start"));
            visitor.visit_mut(&mut self.end, &Meta::field("end"));
        }

        fn variant_name(&self) -> &str {
            unreachable!()
        }

        fn variant_idx(&self) -> usize {
            unreachable!()
        }
    }
}
