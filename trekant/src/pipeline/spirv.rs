use ash::vk::{self, ShaderStageFlags};

use thiserror::Error;

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
        for other_set in other.into_iter() {
            let mut found_set = false;
            for set in self.desc_layouts.iter_mut() {
                if set.set_idx == other_set.set_idx {
                    found_set = true;

                    for other_binding in other_set.bindings.iter() {
                        let mut found_binding = false;
                        for binding in set.bindings.iter_mut() {
                            if binding.binding == other_binding.binding {
                                found_binding = true;
                                assert_eq!(
                                    binding.descriptor_type, other_binding.descriptor_type,
                                    "Descriptor set mismatch in binding {} in set {}",
                                    binding.binding, set.set_idx
                                );
                                assert_eq!(
                                    binding.descriptor_count,
                                    other_binding.descriptor_count
                                );
                                assert_eq!(
                                    binding.p_immutable_samplers,
                                    other_binding.p_immutable_samplers
                                );
                                binding.stage_flags |= other_binding.stage_flags;
                            }
                        }

                        if !found_binding {
                            set.bindings.push(*other_binding);
                        }
                    }
                }
            }

            if !found_set {
                self.desc_layouts.push(other_set);
            }
        }
    }

    fn merge_push_constants(&mut self, mut constants: Vec<vk::PushConstantRange>) {
        for pc_range in self.push_constants.iter() {
            for incoming in constants.iter_mut() {
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
    #[error("Couldn't parse spirv: {0}")]
    Parsing(#[from] rspirv_reflect::ReflectError),
}

fn to_vk_descriptor_type(refl_desc_ty: rspirv_reflect::DescriptorType) -> vk::DescriptorType {
    // rspriv-reflect guarantees bit-exact repr with ash/vk
    vk::DescriptorType::from_raw(refl_desc_ty.0.try_into().unwrap())
}

fn to_descriptor_count(count: rspirv_reflect::BindingCount) -> u32 {
    use rspirv_reflect::BindingCount as Count;

    match count {
        Count::One => 1,
        Count::StaticSized(v) => v.try_into().unwrap(),
        Count::Unbounded => unimplemented!(),
    }
}

pub fn parse_spirv(
    spv_data: &[u32],
    stage: ShaderStageFlags,
) -> Result<ReflectionData, SpirvError> {
    let spv_data: &[u8] = bytemuck::cast_slice(spv_data);
    let module = rspirv_reflect::Reflection::new_from_spirv(spv_data)?;
    let desc_sets = module.get_descriptor_sets()?;
    let mut desc_layouts = Vec::with_capacity(desc_sets.len());
    for (set_idx, desc_set) in desc_sets.into_iter() {
        log::trace!("Found descriptor set: {set_idx}");
        let mut bindings = Vec::with_capacity(desc_set.len());
        for (bind_idx, descriptor_info) in desc_set.into_iter() {
            let binding = vk::DescriptorSetLayoutBinding {
                binding: bind_idx,
                descriptor_type: to_vk_descriptor_type(descriptor_info.ty),
                descriptor_count: to_descriptor_count(descriptor_info.binding_count),
                stage_flags: stage,
                p_immutable_samplers: std::ptr::null(),
            };

            log::trace!("\tdesc set {set_idx}: {binding:?}",);
            bindings.push(binding);
        }

        log::trace!("Created bindings:");
        for b in &bindings {
            log::trace!("\t{b:?}");
        }

        desc_layouts.push(DescriptorSetLayoutData {
            set_idx: set_idx as usize,
            bindings,
        });
    }

    let pc = module.get_push_constant_range()?;
    let mut push_constants = Vec::new();
    if let Some(pc) = pc {
        push_constants.push(vk::PushConstantRange {
            stage_flags: stage,
            offset: pc.offset,
            size: pc.size,
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

    static ARRAY_TEX_SPV_FRAG: &[u32] = inline_spirv::inline_spirv!(
        r"
        #version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(set = 0, binding = 0) uniform sampler2D texs[8];

        layout(location = 0) in vec3 fragColor;
        layout(location = 1) in vec2 fragTexCoord;

        layout(location = 0) out vec4 outColor;

        void main() {
            outColor = texture(texs[0], fragTexCoord).xxyy * vec4(fragColor, 1.0) + vec4(fragTexCoord, 1.0, 0.0);
        }
    ",
        frag
    );

    static STORAGE_BUFFER_SPV_FRAG: &[u32] = inline_spirv::inline_spirv!(
        r"
        #version 450
        #extension GL_ARB_separate_shader_objects : enable

        layout(std140, set = 0, binding = 0) readonly buffer WorldToShadow {
            mat4 data[];
        } world_to_shadow;

        // layout(location = 0) in vec3 fragColor;
        // layout(location = 1) in vec2 fragTexCoord;

        // layout(location = 0) out vec4 outColor;

        void main() {
            // outColor = texture(texs[0], fragTexCoord).xxyy * vec4(fragColor, 1.0) + vec4(fragTexCoord, 1.0, 0.0);
        }
    ",
        frag
    );

    use super::*;

    #[test]
    fn parse_vert_descriptor_set_layout() {
        let refl_data =
            parse_spirv(UBO_SPV_VERT, vk::ShaderStageFlags::VERTEX).expect("Failed to parse!");

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
        let refl_data =
            parse_spirv(UBO_SPV_FRAG, vk::ShaderStageFlags::FRAGMENT).expect("Failed to parse!");

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
        let res = parse_spirv(PUSH_CONSTANT_SPV_FRAG, vk::ShaderStageFlags::FRAGMENT)
            .expect("Failed to parse!");
        assert_eq!(res.desc_layouts.len(), 0);
        assert_eq!(res.push_constants.len(), 1);

        let pc0 = res.push_constants[0];
        assert_eq!(pc0.offset, 0);
        assert_eq!(pc0.size, 128);
        assert_eq!(pc0.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn parse_array_of_textures() {
        let res = parse_spirv(ARRAY_TEX_SPV_FRAG, vk::ShaderStageFlags::FRAGMENT)
            .expect("Failed to parse!");
        assert_eq!(res.desc_layouts.len(), 1);
        assert_eq!(res.push_constants.len(), 0);

        assert_eq!(res.desc_layouts[0].bindings.len(), 1);
        assert_eq!(res.desc_layouts[0].set_idx, 0);

        let binding: vk::DescriptorSetLayoutBinding = res.desc_layouts[0].bindings[0];

        assert_eq!(
            binding.descriptor_type,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        );
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.descriptor_count, 8);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn parse_storage_buffer() {
        let res = parse_spirv(STORAGE_BUFFER_SPV_FRAG, vk::ShaderStageFlags::FRAGMENT)
            .expect("Failed to parse!");
        assert_eq!(res.desc_layouts.len(), 1);
        assert_eq!(res.push_constants.len(), 0);

        assert_eq!(res.desc_layouts[0].bindings.len(), 1);
        assert_eq!(res.desc_layouts[0].set_idx, 0);

        let binding: vk::DescriptorSetLayoutBinding = res.desc_layouts[0].bindings[0];

        assert_eq!(binding.descriptor_type, vk::DescriptorType::STORAGE_BUFFER);
        assert_eq!(binding.binding, 0);
        assert_eq!(binding.descriptor_count, 1);
        assert_eq!(binding.stage_flags, vk::ShaderStageFlags::FRAGMENT);
    }

    #[test]
    fn merge_descriptor_set_layout() {
        let mut res = ReflectionData::new();
        res.merge(
            parse_spirv(UBO_SPV_VERT, vk::ShaderStageFlags::VERTEX).expect("Failed to parse!"),
        );
        res.merge(
            parse_spirv(UBO_SPV_FRAG, vk::ShaderStageFlags::FRAGMENT).expect("Failed to parse!"),
        );
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
