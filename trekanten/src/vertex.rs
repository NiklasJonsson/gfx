use ash::vk;

// TODO: Cleanup traits to use this
#[derive(Debug, Clone)]
pub struct VertexFormat {
    pub binding_description: Vec<vk::VertexInputBindingDescription>,
    pub attribute_description: Vec<vk::VertexInputAttributeDescription>,
}

pub trait VertexDefinition {
    fn binding_description() -> Vec<vk::VertexInputBindingDescription>;
    fn attribute_description() -> Vec<vk::VertexInputAttributeDescription>;
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
