use ash::vk;

use thiserror::Error;

use spirv_reflect::types::descriptor::ReflectDescriptorType;
use spirv_reflect::types::variable::ReflectShaderStageFlags;
use spirv_reflect::ShaderModule;

#[derive(Debug, Default)]
pub struct ReflectionData {
    pub desc_layouts: Vec<DescriptorSetLayoutData>,
    pub push_constants: Vec<vk::PushConstantRange>,
}

impl ReflectionData {
    pub fn new() -> Self {
        Self {
            desc_layouts: Vec::new(),
            push_constants: Vec::new(),
        }
    }

    pub fn merge_layouts(&mut self, other: Vec<DescriptorSetLayoutData>) {
        for mut ol in other.into_iter() {
            let mut found = false;
            for l in self.desc_layouts.iter_mut() {
                if ol.set_idx == l.set_idx {
                    l.bindings.append(&mut ol.bindings);
                    found = true;
                    break;
                }
            }

            if !found {
                self.desc_layouts.push(ol);
            }
        }
    }

    fn merge_push_constants(&mut self, mut constants: Vec<vk::PushConstantRange>) {
        for pc_range in self.push_constants.iter() {
            for incoming in constants.iter() {
                if pc_range.offset + pc_range.size > incoming.offset {
                    log::warn!("Found overlapping push constant range");
                    log::warn!("{:?} ends after {:?} begins", pc_range, incoming);
                }
            }
        }

        self.push_constants.append(&mut constants);
    }

    pub fn merge(&mut self, other: ReflectionData) {
        let Self {
            desc_layouts,
            push_constants,
        } = other;
        self.merge_layouts(desc_layouts);
        self.merge_push_constants(push_constants);
    }
}

#[derive(Debug)]
pub struct DescriptorSetLayoutData {
    pub set_idx: usize,
    pub bindings: Vec<vk::DescriptorSetLayoutBinding>,
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

pub fn log_bindings(bindings: &[spirv_reflect::types::descriptor::ReflectDescriptorBinding]) {
    log::trace!("With {} bindings", bindings.len());
    for b in bindings.iter() {
        log::trace!("\tname: {}", b.name);
        log::trace!("\tbinding: {}", b.binding);
        log::trace!("\tset: {}", b.set);
        log::trace!("\tdescriptor type: {:?}", b.descriptor_type);
        log::trace!("\tresource type: {:?}", b.resource_type);
        log::trace!(
            "\ttype name: {:?}",
            b.type_description.as_ref().map(|x| &x.type_name)
        );
    }
}

pub fn parse_spirv(spv_data: &[u32]) -> Result<ReflectionData, SpirvError> {
    let module = ShaderModule::load_u32_data(spv_data).map_err(SpirvError::Loading)?;
    let desc_sets = module
        .enumerate_descriptor_sets(None)
        .map_err(SpirvError::Parsing)?;
    let stage_flags = map_shader_stage_flags(&module.get_shader_stage());
    let mut desc_layouts = Vec::with_capacity(desc_sets.len());
    for refl_desc_set in desc_sets.iter() {
        let set_idx = refl_desc_set.set;
        log::trace!("Found descriptor set: {}", set_idx);
        log_bindings(&refl_desc_set.bindings);

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = refl_desc_set
            .bindings
            .iter()
            .map(|refl_binding| vk::DescriptorSetLayoutBinding {
                binding: refl_binding.binding,
                descriptor_type: map_descriptor_type(&refl_binding.descriptor_type),
                descriptor_count: 1,
                stage_flags,
                ..Default::default()
            })
            .collect();

        log::trace!("Created bindings:");
        for b in &bindings {
            log::trace!("\t{:?}", b);
        }

        desc_layouts.push(DescriptorSetLayoutData {
            set_idx: set_idx as usize,
            bindings,
        })
    }

    let pc_blocks = module
        .enumerate_push_constant_blocks(None)
        .map_err(SpirvError::Parsing)?;

    let mut push_constants = Vec::with_capacity(pc_blocks.len());
    for pc_block in pc_blocks.iter() {
        log::trace!("Found push constant block:");
        log::trace!("\tname: {}", pc_block.name);
        log::trace!("\toffset: {}", pc_block.offset);
        log::trace!("\tsize: {}", pc_block.size);
        assert_eq!(pc_block.size, pc_block.padded_size);
        assert_eq!(pc_block.offset, pc_block.absolute_offset);

        push_constants.push(vk::PushConstantRange {
            stage_flags,
            offset: pc_block.offset,
            size: pc_block.size,
        });
    }

    Ok(ReflectionData {
        desc_layouts,
        push_constants,
    })
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

    static PUSH_CONSTANT_SPV_FRAG: &[u32] = inline_spirv::inline_spirv!(
        r"
        #version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(push_constant) uniform Model {
            mat4 model;
            mat4 model_it; // inverse transpose of model matrix
        } model_ubo;

        layout(location = 0) in vec3 fragColor;
        layout(location = 1) in vec2 fragTexCoord;

        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = model_ubo.model * vec4(fragColor, 1.0) + model_ubo.model_it * vec4(fragTexCoord, 1.0, 0.0);
        }
    ",
        frag
    );

    use super::*;

    #[test]
    fn parse_vert_descriptor_set_layout() {
        let refl_data = parse_spirv(UBO_SPV_VERT).expect("Failed to parse!");

        assert_eq!(refl_data.push_constants.len(), 0);
        let res = refl_data.desc_layouts;
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
        let refl_data = parse_spirv(UBO_SPV_FRAG).expect("Failed to parse!");

        assert_eq!(refl_data.push_constants.len(), 0);
        let res = refl_data.desc_layouts;
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
    fn parse_frag_push_constant() {
        let res = parse_spirv(PUSH_CONSTANT_SPV_FRAG).expect("Failed to parse!");
        assert_eq!(res.desc_layouts.len(), 0);
        assert_eq!(res.push_constants.len(), 1);

        let pc0 = res.push_constants[0];
        assert_eq!(pc0.offset, 0);
        assert_eq!(pc0.size, 128);
        assert_eq!(pc0.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn merge_descriptor_set_layout() {
        let mut res = ReflectionData::new();
        res.merge(parse_spirv(UBO_SPV_VERT).expect("Failed to parse!"));
        res.merge(parse_spirv(UBO_SPV_FRAG).expect("Failed to parse!"));
        let layouts = res.desc_layouts;
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
