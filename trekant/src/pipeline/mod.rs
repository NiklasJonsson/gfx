use ash::vk;

use derive_builder::Builder;

use std::ffi::CStr;
use std::fmt::Debug;

use crate::backend::render_pass::RenderPass;
use crate::device::HasVkDevice;
use crate::device::VkDeviceHandle;
use crate::resource::CachedStorage;
use crate::vertex::VertexFormat;

mod error;
mod spirv;

pub use error::PipelineError;
use spirv::{parse_spirv, ReflectionData};

bitflags::bitflags! {
    pub struct ShaderStage: u8 {
        const VERTEX = 0b1;
        const FRAGMENT = 0b10;
    }
}

fn hash<T: std::hash::Hash>(t: T) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut state = DefaultHasher::new();
    t.hash(&mut state);
    state.finish()
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(s: ShaderStage) -> Self {
        let mut out = vk::ShaderStageFlags::default();

        if s.contains(ShaderStage::VERTEX) {
            out |= vk::ShaderStageFlags::VERTEX;
        }

        if s.contains(ShaderStage::FRAGMENT) {
            out |= vk::ShaderStageFlags::FRAGMENT;
        }

        out
    }
}

struct ShaderModule {
    vk_device: VkDeviceHandle,
    vk_shader_module: vk::ShaderModule,
}

impl std::fmt::Debug for ShaderModule {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_tuple("ShaderModule")
            .field(&self.vk_shader_module)
            .finish()
    }
}

impl std::ops::Drop for ShaderModule {
    fn drop(&mut self) {
        log::trace!("Dropping shader module {:?}", self);
        unsafe {
            self.vk_device
                .destroy_shader_module(self.vk_shader_module, None);
        }
    }
}

impl ShaderModule {
    pub fn new<D: HasVkDevice>(device: &D, spirv: &[u32]) -> Result<Self, PipelineError> {
        log::trace!("Creating shader module from data (hash) {}", hash(spirv));
        let info = vk::ShaderModuleCreateInfo::builder().code(spirv);

        let vk_device = device.vk_device();

        let vk_shader_module = unsafe {
            vk_device
                .create_shader_module(&info, None)
                .map_err(|e| PipelineError::VulkanObjectCreation(e, "Shader module"))?
        };

        log::trace!("Created shader module: {:?}", vk_shader_module);

        Ok(Self {
            vk_device,
            vk_shader_module,
        })
    }
}

pub trait Pipeline {
    const BIND_POINT: vk::PipelineBindPoint;

    fn vk_pipeline(&self) -> &vk::Pipeline;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriangleCulling {
    Front,
    Back,
    None,
}

impl Default for TriangleCulling {
    fn default() -> Self {
        Self::Back
    }
}

impl From<TriangleCulling> for vk::CullModeFlags {
    fn from(tc: TriangleCulling) -> Self {
        match tc {
            TriangleCulling::Front => Self::FRONT,
            TriangleCulling::Back => Self::BACK,
            TriangleCulling::None => Self::NONE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    TriangleList,
    LineStrip,
}

impl Default for PrimitiveTopology {
    fn default() -> Self {
        Self::TriangleList
    }
}

impl From<PrimitiveTopology> for vk::PrimitiveTopology {
    fn from(pt: PrimitiveTopology) -> Self {
        match pt {
            PrimitiveTopology::LineStrip => Self::LINE_STRIP,
            PrimitiveTopology::TriangleList => Self::TRIANGLE_LIST,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriangleWinding {
    Clockwise,
    CounterClockwise,
}

impl Default for TriangleWinding {
    fn default() -> Self {
        Self::CounterClockwise
    }
}

impl From<TriangleWinding> for vk::FrontFace {
    fn from(tw: TriangleWinding) -> Self {
        match tw {
            TriangleWinding::Clockwise => Self::CLOCKWISE,
            TriangleWinding::CounterClockwise => Self::COUNTER_CLOCKWISE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendState {
    Enabled,
    Disabled,
}

impl Default for BlendState {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DepthTest {
    Enabled,
    Disabled,
}

impl Default for DepthTest {
    fn default() -> Self {
        Self::Enabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

impl Default for PolygonMode {
    fn default() -> Self {
        Self::Fill
    }
}

impl From<PolygonMode> for vk::PolygonMode {
    fn from(pm: PolygonMode) -> Self {
        match pm {
            PolygonMode::Fill => vk::PolygonMode::FILL,
            PolygonMode::Line => vk::PolygonMode::LINE,
            PolygonMode::Point => vk::PolygonMode::POINT,
        }
    }
}

pub struct GraphicsPipeline {
    vk_device: VkDeviceHandle,
    vk_pipeline: vk::Pipeline,
    vk_pipeline_layout: vk::PipelineLayout,
    vk_descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl Pipeline for GraphicsPipeline {
    const BIND_POINT: vk::PipelineBindPoint = vk::PipelineBindPoint::GRAPHICS;

    fn vk_pipeline(&self) -> &vk::Pipeline {
        &self.vk_pipeline
    }
}

impl std::ops::Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.vk_device.destroy_pipeline(self.vk_pipeline, None);

            self.vk_device
                .destroy_pipeline_layout(self.vk_pipeline_layout, None);
            for &dset_layout in self.vk_descriptor_set_layouts.iter() {
                self.vk_device
                    .destroy_descriptor_set_layout(dset_layout, None);
            }
        }
    }
}

struct PipelineCreationInfo {
    create_info: vk::PipelineShaderStageCreateInfo,
    shader_module: ShaderModule,
}

const DEFAULT_SPV_ENTRY_NAME_NULLED: &[u8] = b"main\0";

impl GraphicsPipeline {
    pub fn vk_descriptor_set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.vk_descriptor_set_layouts
    }

    pub fn vk_pipeline_layout(&self) -> &vk::PipelineLayout {
        &self.vk_pipeline_layout
    }

    fn shader<D: HasVkDevice>(
        device: &D,
        refl_data: &mut ReflectionData,
        desc: &ShaderDescriptor,
        stage: vk::ShaderStageFlags,
    ) -> Result<PipelineCreationInfo, PipelineError> {
        log::trace!("Creating shader ({desc:?}) for pipeline");
        let name = CStr::from_bytes_with_nul(DEFAULT_SPV_ENTRY_NAME_NULLED).unwrap();

        let shader_module = ShaderModule::new(device, &desc.spirv_code)?;
        let create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(shader_module.vk_shader_module)
            .name(name)
            .build();

        let new_refl_data = parse_spirv(&desc.spirv_code).map_err(PipelineError::Reflection)?;

        refl_data.merge(new_refl_data);

        Ok(PipelineCreationInfo {
            create_info,
            shader_module,
        })
    }

    fn create<D: HasVkDevice>(
        device: &D,
        render_pass: &RenderPass,
        desc: &GraphicsPipelineDescriptor,
    ) -> Result<Self, PipelineError> {
        let mut reflection_data = ReflectionData::new();
        let PipelineCreationInfo {
            shader_module: _vert_module,
            create_info: vert_create_info,
        } = Self::shader(
            device,
            &mut reflection_data,
            &desc.vert,
            vk::ShaderStageFlags::VERTEX,
        )?;
        let frag = desc
            .frag
            .as_ref()
            .map(|frag| {
                Self::shader(
                    device,
                    &mut reflection_data,
                    frag,
                    vk::ShaderStageFlags::FRAGMENT,
                )
            })
            .transpose()?;
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(desc.vertex_format.vk_binding_description())
            .vertex_attribute_descriptions(desc.vertex_format.vk_attribute_description());

        let vk_device = device.vk_device();
        // TODO(perf): allocation here
        let mut stages = vec![vert_create_info];
        if let Some(PipelineCreationInfo { create_info, .. }) = &frag {
            stages.push(*create_info);
        }

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(desc.primitive_topology.into())
            .primitive_restart_enable(false);

        let raster_state_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(desc.polygon_mode.into())
            .line_width(1.0)
            .cull_mode(desc.culling.into())
            .front_face(desc.winding.into())
            .depth_bias_enable(false);

        let msaa_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(render_pass.msaa_sample_count());

        let color_blend_attach_info = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attach_info = match desc.blend_state {
            BlendState::Disabled => color_blend_attach_info.blend_enable(false),
            BlendState::Enabled => color_blend_attach_info
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD),
        };

        let attachments = [*color_blend_attach_info];
        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&attachments);

        let mut descriptor_set_layouts = Vec::with_capacity(reflection_data.desc_layouts.len());
        for dset in reflection_data.desc_layouts.iter() {
            let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&dset.bindings);

            let dset_layout = unsafe {
                vk_device
                    .create_descriptor_set_layout(&info, None)
                    .map_err(|e| PipelineError::VulkanObjectCreation(e, "Descriptor set layout"))?
            };

            descriptor_set_layouts.push(dset_layout);
        }

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&reflection_data.push_constants);

        let pipeline_layout = unsafe {
            vk_device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| PipelineError::VulkanObjectCreation(e, "Pipeline layout"))?
        };

        log::trace!("Created pipeline layout {:?}", pipeline_layout);

        let depth_stencil = match desc.depth_testing {
            DepthTest::Disabled => vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(false)
                .depth_write_enable(false),
            DepthTest::Enabled => vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false),
        };

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(0.0)
            .height(0.0)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor_extent = vk::Extent2D {
            width: 0,
            height: 0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: scissor_extent,
        };

        let viewports = [*viewport];
        let scissors = [scissor];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let g_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&raster_state_info)
            .multisample_state(&msaa_info)
            .color_blend_state(&color_blend_state_info)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(*render_pass.vk_render_pass())
            .subpass(0);

        let create_infos = [*g_pipeline_info];

        // TODO: Use the cache
        let vk_pipelines_result = unsafe {
            vk_device.create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
        };
        // According to: https://renderdoc.org/vkspec_chunked/chap10.html#pipelines-multiple
        // Implementations will attempt to create as many pipelines as possible, but if any fail, we really want to exit anyway.

        let pipelines = vk_pipelines_result
            .map_err(|(_vec, e)| PipelineError::VulkanObjectCreation(e, "Pipeline(s)"))?;

        assert_eq!(pipelines.len(), 1, "Expected single pipeline");

        let vk_pipeline = pipelines[0];

        Ok(GraphicsPipeline {
            vk_device,
            vk_pipeline,
            vk_pipeline_layout: pipeline_layout,
            vk_descriptor_set_layouts: descriptor_set_layouts,
        })
    }
}

#[derive(Clone)]
pub struct ShaderDescriptor {
    pub debug_name: Option<String>,
    pub spirv_code: Vec<u32>,
}

impl Debug for ShaderDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "ShaderDescriptor {{\
    debug_name: {} \
    spirv_code: {{ start: {:p}, size: {}, hash: {:#X} }}
}}",
            self.debug_name
                .as_ref()
                .map(String::as_str)
                .unwrap_or("N/A"),
            self.spirv_code.as_ptr(),
            self.spirv_code.len(),
            hash(&self.spirv_code)
        ))
    }
}

impl PartialEq for ShaderDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.spirv_code == other.spirv_code
    }
}

impl Eq for ShaderDescriptor {}

impl std::hash::Hash for ShaderDescriptor {
    fn hash<H>(&self, state: &mut H)
    where
        H: std::hash::Hasher,
    {
        self.spirv_code.hash(state);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Builder)]
#[builder(pattern = "owned")]
#[builder(build_fn(name = "generated_build"))]
pub struct GraphicsPipelineDescriptor {
    pub vert: ShaderDescriptor,
    #[builder(setter(strip_option), default)]
    pub frag: Option<ShaderDescriptor>,
    pub vertex_format: VertexFormat,
    #[builder(default)]
    pub culling: TriangleCulling,
    #[builder(default)]
    pub winding: TriangleWinding,
    #[builder(default)]
    pub blend_state: BlendState,
    #[builder(default)]
    pub depth_testing: DepthTest,
    #[builder(default)]
    pub polygon_mode: PolygonMode,
    #[builder(default)]
    pub primitive_topology: PrimitiveTopology,
}

impl GraphicsPipelineDescriptorBuilder {
    pub fn build(self) -> Result<GraphicsPipelineDescriptor, PipelineError> {
        self.generated_build()
            .map_err(PipelineError::GraphicsPipelineBuilder)
    }
}

impl GraphicsPipelineDescriptor {
    pub fn builder() -> GraphicsPipelineDescriptorBuilder {
        GraphicsPipelineDescriptorBuilder::default()
    }

    pub fn create<D: HasVkDevice>(
        &self,
        device: &D,
        render_pass: &RenderPass,
    ) -> Result<GraphicsPipeline, PipelineError> {
        GraphicsPipeline::create(device, render_pass, self)
    }
}

pub type GraphicsPipelines = CachedStorage<GraphicsPipelineDescriptor, GraphicsPipeline>;
