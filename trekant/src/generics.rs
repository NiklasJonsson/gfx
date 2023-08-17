pub struct Cond<const COND: bool> {}

pub trait True {}
pub trait False {}

impl True for Cond<true> {}
impl False for Cond<false> {}
