use crate::util;
use ash::vk;
use std::hash::{Hash, Hasher};

// TODO: vk agnostic
#[derive(Debug, Clone)]
pub struct VertexFormat {
    binding_description: Vec<vk::VertexInputBindingDescription>,
    attribute_description: Vec<vk::VertexInputAttributeDescription>,
    size: u32,
}

impl std::fmt::Display for VertexFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "VertexFormat (size: {}) {{", self.size)?;
        for (i, b) in self.binding_description.iter().enumerate() {
            writeln!(
                f,
                "\tbindings[{}] = (binding: {}, rate: {:?}, stride: {})",
                i, b.binding, b.input_rate, b.stride
            )?;
        }
        for a in self.attribute_description.iter() {
            let s = match a.format {
                vk::Format::R32G32B32A32_SFLOAT => "vec4",
                vk::Format::R32G32B32_SFLOAT => "vec3",
                vk::Format::R32G32_SFLOAT => "vec2",
                vk::Format::R32_SFLOAT => "float",
                _ => unimplemented!("Unsupported vertex format"),
            };
            writeln!(f, "\t[{}][{}](+{}) {}", a.binding, a.location, a.offset, s)?;
        }
        writeln!(f, "}}")
    }
}

// TODO: Autogen eq/hash
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
    pub fn vk_binding_description(&self) -> &[vk::VertexInputBindingDescription] {
        &self.binding_description
    }

    pub fn vk_attribute_description(&self) -> &[vk::VertexInputAttributeDescription] {
        &self.attribute_description
    }

    pub fn size(&self) -> u32 {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl VertexFormat {
    pub fn empty() -> Self {
        Self {
            binding_description: Vec::new(),
            attribute_description: Vec::new(),
            size: 0,
        }
    }

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
    fn format() -> VertexFormat;
}
