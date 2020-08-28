use crate::util;
use ash::vk;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct VertexFormat {
    binding_description: Vec<vk::VertexInputBindingDescription>,
    attribute_description: Vec<vk::VertexInputAttributeDescription>,
    size: u32,
}
impl Eq for VertexFormat {}
impl PartialEq for VertexFormat {
    fn eq(&self, o: &Self) -> bool {
        if self.binding_description.len() != o.binding_description.len()
            || self.attribute_description.len() != o.attribute_description.len()
            || self.size != o.size
        {
            return false;
        }

        self.binding_description
            .iter()
            .zip(o.binding_description.iter())
            .fold(true, |acc, (a, b)| {
                acc && a.binding == b.binding
                    && a.stride == b.stride
                    && a.input_rate == b.input_rate
            })
            && self
                .attribute_description
                .iter()
                .zip(o.attribute_description.iter())
                .fold(true, |acc, (a, b)| {
                    acc && a.location == b.location
                        && a.binding == b.binding
                        && a.format == b.format
                        && a.offset == b.offset
                })
    }
}

// TODO: Autgen this
impl Hash for VertexFormat {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for b in self.binding_description.iter() {
            b.binding.hash(state);
            b.stride.hash(state);
            b.input_rate.hash(state);
        }
        for a in self.attribute_description.iter() {
            a.location.hash(state);
            a.binding.hash(state);
            a.format.hash(state);
            a.offset.hash(state);
        }
        self.size.hash(state);
    }
}

impl VertexFormat {
    pub fn vk_binding_description(&self) -> &vk::VertexInputBindingDescription {
        &self.binding_description[0]
    }

    pub fn vk_attribute_descriptions(&self) -> &[vk::VertexInputAttributeDescription] {
        &self.attribute_description
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    fn empty() -> Self {
        Self {
            binding_description: Vec::new(),
            attribute_description: Vec::new(),
            size: 0,
        }
    }
}

impl VertexFormat {
    pub fn builder() -> VertexFormatBuilder {
        VertexFormatBuilder::new()
    }
}

pub struct VertexFormatBuilder {
    format: VertexFormat,
}

impl VertexFormatBuilder {
    fn new() -> Self {
        Self {
            format: VertexFormat::empty(),
        }
    }

    pub fn add_attribute(mut self, format: util::Format) -> Self {
        let size = format.size();
        self.format
            .attribute_description
            .push(vk::VertexInputAttributeDescription {
                binding: 0,
                location: self.format.attribute_description.len() as u32,
                format: format.into(),
                offset: self.format.size,
            });
        self.format.size += size;
        self
    }

    pub fn build(mut self) -> VertexFormat {
        self.format.binding_description = vec![vk::VertexInputBindingDescription {
            binding: 0,
            stride: self.format.size,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        self.format
    }
}

pub trait VertexDefinition {
    fn binding_description() -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_description() -> Vec<vk::VertexInputAttributeDescription>;
    fn format() -> VertexFormat {
        let attribute_description = Self::attribute_description();
        let size = attribute_description
            .iter()
            .fold(0, |acc, d| acc + util::Format::from(d.format).size());
        VertexFormat {
            binding_description: Self::binding_description(),
            attribute_description: attribute_description,
            size,
        }
    }
}

pub trait VertexSource {
    fn binding_description(&self) -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_description(&self) -> Vec<vk::VertexInputAttributeDescription>;
}

impl<V: VertexDefinition> VertexSource for Vec<V> {
    fn binding_description(&self) -> Vec<vk::VertexInputBindingDescription> {
        V::binding_description()
    }

    fn attribute_description(&self) -> Vec<vk::VertexInputAttributeDescription> {
        V::attribute_description()
    }
}
