use std::marker::PhantomData;
use std::rc::Rc;

pub trait Asset {}

pub type AssetID = u32;

pub struct Handle<A: Asset> {
    id: Rc<AssetID>,
    asset_type: PhantomData<A>,
}

pub struct Storage<A: Asset> {
    data: Vec<A>,
    handles: Vec<Rc<AssetID>>,
}
