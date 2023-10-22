use crate::util::ByteBuffer;

use std::sync::Arc;

/// A utility type for descriptors to hold untyped data
#[derive(Debug)]
pub enum DescriptorData<'a> {
    Owned(ByteBuffer),
    Shared(Arc<ByteBuffer>),
    Borrowed(&'a [u8]),
}

impl<'a> DescriptorData<'a> {
    pub fn from_vec(v: Vec<u8>) -> Self {
        Self::Owned(unsafe { ByteBuffer::from_vec(v) })
    }
}

impl<'a> DescriptorData<'a> {
    pub fn data(&self) -> &[u8] {
        match self {
            Self::Owned(buf) => &buf,
            Self::Borrowed(buf) => buf,
            Self::Shared(buf) => *&buf,
        }
    }
}

impl<'a> Clone for DescriptorData<'a> {
    fn clone(&self) -> Self {
        match self {
            DescriptorData::Borrowed(data) => DescriptorData::Borrowed(data),
            DescriptorData::Owned(data) => DescriptorData::Owned(data.clone()),
            DescriptorData::Shared(data) => DescriptorData::Shared(Arc::clone(data)),
        }
    }
}
