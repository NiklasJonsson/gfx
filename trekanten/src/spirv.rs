use ash::vk;

use thiserror::Error;

use spirv_reflect::types::descriptor::ReflectDescriptorType;
use spirv_reflect::types::variable::ReflectShaderStageFlags;
use spirv_reflect::ShaderModule;

#[derive(Debug)]
pub struct DescriptorSetLayoutData {
    pub set_idx: usize,
    pub bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

#[derive(Debug)]
pub struct DescriptorSetLayouts {
    layouts: Vec<DescriptorSetLayoutData>,
}

impl DescriptorSetLayouts {
    pub fn new() -> Self {
        Self {
            layouts: Vec::new(),
        }
    }

    pub fn with(layouts: Vec<DescriptorSetLayoutData>) -> Self {
        Self { layouts }
    }

    pub fn append(&mut self, other: DescriptorSetLayouts) {
        dbg!("Before append", &self);
        dbg!("Before append", &other);
        for mut ol in other.layouts.into_iter() {
            let mut found = false;
            for l in self.layouts.iter_mut() {
                if ol.set_idx == l.set_idx {
                    l.bindings.append(&mut ol.bindings);
                    found = true;
                    break;
                }
            }

            if !found {
                self.layouts.push(ol);
            }
        }

        dbg!("After append", self);
    }

    pub fn layouts(&self) -> impl Iterator<Item = &DescriptorSetLayoutData> {
        self.layouts.iter()
    }

    pub fn len(&self) -> usize {
        self.layouts.len()
    }
}

#[derive(Debug, Error)]
pub enum SpirvError {
    #[error("Couldn't load spirv: {0}")]
    Loading(&'static str),
    #[error("Couldn't parse spirv: {0}")]
    Parsing(&'static str),
}

fn map_shader_stage_flags(refl_stage: &ReflectShaderStageFlags) -> vk::ShaderStageFlags {
    match *refl_stage {
        ReflectShaderStageFlags::VERTEX => vk::ShaderStageFlags::VERTEX,
        ReflectShaderStageFlags::FRAGMENT => vk::ShaderStageFlags::FRAGMENT,
        _ => unimplemented!("Unsupported shader stage: {:?}", refl_stage),
    }
}

fn map_descriptor_type(refl_desc_ty: &ReflectDescriptorType) -> vk::DescriptorType {
    match *refl_desc_ty {
        ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        ReflectDescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        _ => unimplemented!("Unsupported descriptor type: {:?}", refl_desc_ty),
    }
}

pub fn parse_descriptor_sets(spv_data: &[u32]) -> Result<DescriptorSetLayouts, SpirvError> {
    let module = ShaderModule::load_u32_data(spv_data).map_err(SpirvError::Loading)?;
    let desc_sets = module
        .enumerate_descriptor_sets(None)
        .map_err(SpirvError::Parsing)?;
    let shader_stage = map_shader_stage_flags(&module.get_shader_stage());
    let mut ret = Vec::with_capacity(desc_sets.len());
    for refl_desc_set in desc_sets.iter() {
        let set_idx = refl_desc_set.set;
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = refl_desc_set
            .bindings
            .iter()
            .map(|refl_binding| vk::DescriptorSetLayoutBinding {
                binding: refl_binding.binding,
                descriptor_type: map_descriptor_type(&refl_binding.descriptor_type),
                descriptor_count: 1,
                stage_flags: shader_stage,
                ..Default::default()
            })
            .collect();
        log::trace!("Found descriptor set: {}", set_idx);
        log::trace!("With {} bindings", bindings.len());
        for b in &bindings {
            log::trace!("\t{:?}", b);
        }

        ret.push(DescriptorSetLayoutData {
            set_idx: set_idx as usize,
            bindings,
        })
    }

    Ok(DescriptorSetLayouts::with(ret))
}

#[cfg(test)]
mod tests {
    static UBO_SPV_VERT: &[u32] = inline_spirv::inline_spirv!(
        r"
        #version 450 core
        layout(set = 0, binding = 0) uniform UniformBufferObject {
            mat4 model;
            mat4 view;
            mat4 proj;
        } ubo;

        void main() {}
    ",
        vert
    );

    static UBO_SPV_FRAG: &[u32] = inline_spirv::inline_spirv!(
        r"
        #version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(set = 0, binding = 1) uniform sampler2D u_colorMap;

        layout(location = 0) in vec3 fragColor;
        layout(location = 1) in vec2 fragTexCoord;

        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = texture(u_colorMap, fragTexCoord);
        }
    ",
        frag
    );
    use super::*;

    #[test]
    fn parse_vert_descriptor_set_layout() {
        let res = parse_descriptor_sets(UBO_SPV_VERT)
            .expect("Failed to parse!")
            .layouts;
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].bindings.len(), 1);
        assert_eq!(res[0].set_idx, 0);

        let binding: vk::DescriptorSetLayoutBinding = res[0].bindings[0];

        assert_eq!(binding.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.descriptor_count, 1);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::VERTEX);
    }

    #[test]
    fn parse_frag_descriptor_set_layout() {
        let res = parse_descriptor_sets(UBO_SPV_FRAG)
            .expect("Failed to parse!")
            .layouts;
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].bindings.len(), 1);
        assert_eq!(res[0].set_idx, 0);

        let binding: vk::DescriptorSetLayoutBinding = res[0].bindings[0];

        assert_eq!(
            binding.descriptor_type,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        );
        assert_eq!(binding.binding, 1);
        assert_eq!(binding.descriptor_count, 1);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn merge_descriptor_set_layout() {
        let mut res = DescriptorSetLayouts::new();
        res.append(parse_descriptor_sets(UBO_SPV_VERT).expect("Failed to parse!"));
        res.append(parse_descriptor_sets(UBO_SPV_FRAG).expect("Failed to parse!"));
        let layouts = res.layouts;
        assert_eq!(layouts.len(), 1);
        let l = &layouts[0];
        assert_eq!(l.bindings.len(), 2);
        assert_eq!(l.set_idx, 0);

        let binding0: vk::DescriptorSetLayoutBinding = l.bindings[0];
        assert_eq!(binding0.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(binding0.binding, 0);
        assert_eq!(binding0.descriptor_count, 1);
        assert_eq!(binding0.stage_flags, vk::ShaderStageFlags::VERTEX);

        let binding1: vk::DescriptorSetLayoutBinding = l.bindings[1];
        assert_eq!(
            binding1.descriptor_type,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        );
        assert_eq!(binding1.binding, 1);
        assert_eq!(binding1.descriptor_count, 1);
        assert_eq!(binding1.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }
}
