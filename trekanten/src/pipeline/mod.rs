use ash::version::DeviceV1_0;
use ash::vk;

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use crate::device::Device;
use crate::device::HasVkDevice;
use crate::device::VkDeviceHandle;
use crate::render_pass::RenderPass;
use crate::resource::{CachedStorage, Handle, ResourceManager};
use crate::spirv::{parse_descriptor_sets, DescriptorSetLayouts};
use crate::util;
use crate::vertex::{VertexDefinition, VertexFormat};

mod error;
pub use error::PipelineError;

pub enum ShaderStage {
    Vertex,
    Fragment,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(s: ShaderStage) -> Self {
        match s {
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        }
    }
}

struct RawShader {
    pub data: Vec<u32>,
}

fn read_shader_abs<P: AsRef<Path>>(path: P) -> io::Result<RawShader> {
    let path_ref = path.as_ref();
    log::trace!("Reading shader from {}", path_ref.display());
    let mut file = File::open(path_ref)?;
    let words = ash::util::read_spv(&mut file)?;
    Ok(RawShader { data: words })
}

fn read_shader_rel<N: AsRef<Path>>(name: N) -> io::Result<RawShader> {
    let cd = std::env::current_dir()?;
    let path = cd
        .join("trekanten")
        .join("src")
        .join("pipeline")
        .join("shaders")
        .join(name);

    read_shader_abs(path)
}

struct ShaderModule {
    vk_device: VkDeviceHandle,
    vk_shader_module: vk::ShaderModule,
}

impl std::ops::Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.vk_device
                .destroy_shader_module(self.vk_shader_module, None);
        }
    }
}

impl ShaderModule {
    pub fn new(device: &Device, raw: &RawShader) -> Result<Self, PipelineError> {
        let info = vk::ShaderModuleCreateInfo::builder().code(&raw.data);

        let vk_device = device.vk_device();

        let vk_shader_module = unsafe {
            vk_device
                .create_shader_module(&info, None)
                .map_err(|e| PipelineError::VulkanObjectCreation(e, "Shader module"))?
        };

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

impl GraphicsPipeline {
    pub fn builder(device: &Device) -> GraphicsPipelineBuilder {
        GraphicsPipelineBuilder::new(device)
    }

    pub fn vk_descriptor_set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.vk_descriptor_set_layouts
    }

    pub fn vk_pipeline_layout(&self) -> &vk::PipelineLayout {
        &self.vk_pipeline_layout
    }
}

struct PipelineCreationInfo {
    create_info: vk::PipelineShaderStageCreateInfo,
    _shader_module: ShaderModule,
}

struct VertexInputDescription<'a> {
    _binding_description: &'a [vk::VertexInputBindingDescription],
    _attribute_description: &'a [vk::VertexInputAttributeDescription],
    create_info: vk::PipelineVertexInputStateCreateInfo,
}
pub struct GraphicsPipelineBuilder<'a> {
    device: &'a Device,
    entry_name: CString,
    vert: Option<PipelineCreationInfo>,
    frag: Option<PipelineCreationInfo>,
    vertex_input: Option<VertexInputDescription<'a>>,
    viewport_extent: Option<util::Extent2D>,
    render_pass: Option<&'a RenderPass>,
    refl_descriptor_set_layouts: DescriptorSetLayouts,
}

impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new(device: &'a Device) -> Self {
        let entry_name = CString::new("main").expect("CString failed!");
        Self {
            device,
            entry_name,
            vert: None,
            frag: None,
            vertex_input: None,
            render_pass: None,
            viewport_extent: None,
            refl_descriptor_set_layouts: DescriptorSetLayouts::new(),
        }
    }

    fn shader<P: AsRef<Path>>(
        &mut self,
        path: P,
        stage: vk::ShaderStageFlags,
    ) -> Result<PipelineCreationInfo, PipelineError> {
        log::trace!("Reading shader {}", path.as_ref().display());
        let raw = read_shader_rel(path)?;
        let shader_module = ShaderModule::new(self.device, &raw)?;
        let create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(stage)
            .module(shader_module.vk_shader_module)
            .name(&self.entry_name)
            .build();

        let new_desc_sets = parse_descriptor_sets(&raw.data).map_err(PipelineError::Reflection)?;

        self.refl_descriptor_set_layouts.append(new_desc_sets);

        Ok(PipelineCreationInfo {
            create_info,
            _shader_module: shader_module,
        })
    }

    pub fn vertex_shader<P: AsRef<Path>>(mut self, path: P) -> Result<Self, PipelineError> {
        self.vert = Some(self.shader(path, vk::ShaderStageFlags::VERTEX)?);
        Ok(self)
    }

    pub fn fragment_shader<P: AsRef<Path>>(mut self, path: P) -> Result<Self, PipelineError> {
        self.frag = Some(self.shader(path, vk::ShaderStageFlags::FRAGMENT)?);
        Ok(self)
    }

    pub fn vertex_input(
        mut self,
        attribute_description: &'a [vk::VertexInputAttributeDescription],
        binding_description: &'a [vk::VertexInputBindingDescription],
    ) -> Self {
        let create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_description)
            .vertex_attribute_descriptions(&attribute_description)
            .build();

        self.vertex_input = Some(VertexInputDescription {
            _attribute_description: attribute_description,
            _binding_description: binding_description,
            create_info,
        });

        self
    }

    pub fn viewport_extent(mut self, extent: util::Extent2D) -> Self {
        self.viewport_extent = Some(extent);
        self
    }

    pub fn render_pass(mut self, render_pass: &'a RenderPass) -> Self {
        self.render_pass = Some(render_pass);
        self
    }

    pub fn build(self) -> Result<GraphicsPipeline, PipelineError> {
        let vert = self
            .vert
            .ok_or(PipelineError::MissingArg("vertex shader"))?;
        let frag = self
            .frag
            .ok_or(PipelineError::MissingArg("fragment shader"))?;
        let vertex_input = self
            .vertex_input
            .ok_or(PipelineError::MissingArg("vertex description"))?;
        let viewport_extent = self
            .viewport_extent
            .ok_or(PipelineError::MissingArg("viewport extent"))?;
        let render_pass = self
            .render_pass
            .ok_or(PipelineError::MissingArg("render pass"))?;

        let vk_device = self.device.vk_device();
        let stages = [vert.create_info, frag.create_info];

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let raster_state_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let msaa_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(render_pass.msaa_sample_count());

        let color_blend_attach_info = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false);

        let attachments = [*color_blend_attach_info];
        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&attachments);

        let mut descriptor_set_layouts = Vec::with_capacity(self.refl_descriptor_set_layouts.len());
        for dset in self.refl_descriptor_set_layouts.layouts() {
            let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&dset.bindings);

            let dset_layout = unsafe {
                vk_device
                    .create_descriptor_set_layout(&info, None)
                    .map_err(|e| PipelineError::VulkanObjectCreation(e, "Descriptor set layout"))?
            };

            descriptor_set_layouts.push(dset_layout);
        }

        let pces = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 128,
        }];

        // TODO: Parse this
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&pces);

        let pipeline_layout = unsafe {
            vk_device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| PipelineError::VulkanObjectCreation(e, "Pipeline layout"))?
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(viewport_extent.width as f32)
            .height(viewport_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor_extent: vk::Extent2D = viewport_extent.into();

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: scissor_extent,
        };

        let viewports = [*viewport];
        let scissors = [scissor];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let g_pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input.create_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&raster_state_info)
            .multisample_state(&msaa_info)
            .color_blend_state(&color_blend_state_info)
            .depth_stencil_state(&depth_stencil)
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineDescriptor {
    pub vert: PathBuf,
    pub frag: PathBuf,
    pub vertex_format: VertexFormat,
}

impl GraphicsPipelineDescriptor {
    pub fn builder() -> GraphicsPipelineDescriptorBuilder {
        GraphicsPipelineDescriptorBuilder {
            vert: None,
            frag: None,
            vertex_format: VertexFormat::empty(),
        }
    }
}

pub struct GraphicsPipelineDescriptorBuilder {
    vert: Option<PathBuf>,
    frag: Option<PathBuf>,
    vertex_format: VertexFormat,
}

impl GraphicsPipelineDescriptorBuilder {
    pub fn vertex_shader<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.vert = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn fragment_shader<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.frag = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn vertex_type<V>(mut self) -> Self
    where
        V: VertexDefinition,
    {
        self.vertex_format = V::format();
        self
    }

    pub fn build(self) -> Result<GraphicsPipelineDescriptor, PipelineError> {
        let vert = self
            .vert
            .ok_or(PipelineError::MissingArg("vertex shader"))?;
        let frag = self
            .frag
            .ok_or(PipelineError::MissingArg("fragment shader"))?;
        if self.vertex_format.is_empty() {
            return Err(PipelineError::MissingArg("vertex description"));
        }

        Ok(GraphicsPipelineDescriptor {
            vert,
            frag,
            vertex_format: self.vertex_format,
        })
    }
}

#[derive(Default)]
pub struct GraphicsPipelines {
    mat_storage: CachedStorage<GraphicsPipelineDescriptor, GraphicsPipeline>,
}

impl GraphicsPipelines {
    pub fn new() -> Self {
        Self {
            mat_storage: Default::default(),
        }
    }

    fn create_pipeline(
        device: &Device,
        viewport_extent: util::Extent2D,
        render_pass: &RenderPass,
        descriptor: &GraphicsPipelineDescriptor,
    ) -> Result<GraphicsPipeline, PipelineError> {
        GraphicsPipeline::builder(device)
            .vertex_shader(&descriptor.vert)?
            .fragment_shader(&descriptor.frag)?
            .vertex_input(
                &descriptor.vertex_format.vk_attribute_description(),
                descriptor.vertex_format.vk_binding_description(),
            )
            .viewport_extent(viewport_extent)
            .render_pass(render_pass)
            .build()
    }

    pub fn recreate_all(
        &mut self,
        device: &Device,
        viewport_extent: util::Extent2D,
        render_pass: &RenderPass,
    ) -> Result<(), PipelineError> {
        for (desc, pipe) in self.mat_storage.iter_mut() {
            *pipe = Self::create_pipeline(device, viewport_extent, render_pass, desc)?;
        }

        Ok(())
    }

    pub fn create(
        &mut self,
        device: &Device,
        descriptor: GraphicsPipelineDescriptor,
        viewport_extent: util::Extent2D,
        render_pass: &RenderPass,
    ) -> Result<Handle<GraphicsPipeline>, PipelineError> {
        self.mat_storage.create_or_add(descriptor, |desc| {
            Self::create_pipeline(device, viewport_extent, render_pass, &desc)
        })
    }

    pub fn get(&self, h: &Handle<GraphicsPipeline>) -> Option<&GraphicsPipeline> {
        self.mat_storage.get(h)
    }
}

pub fn get_pipeline_for(
    renderer: &mut crate::Renderer,
    mesh: &crate::mesh::Mesh,
    mat: &crate::material::MaterialData,
) -> Handle<GraphicsPipeline> {
    let vertex_format = renderer
        .get_resource(&mesh.vertex_buffer.handle())
        .expect("Invalid handle")
        .format
        .clone();
    match mat {
        crate::material::MaterialData::PBR {
            normal_map,
            base_color_texture,
            metallic_roughness_texture,
            has_vertex_colors,
            ..
        } => {
            let has_nm = normal_map.is_some();
            let has_bc = base_color_texture.is_some();
            let has_mr = metallic_roughness_texture.is_some();
            let desc = if has_nm && has_bc && has_mr && !has_vertex_colors {
                GraphicsPipelineDescriptor {
                    vert: PathBuf::from("vs_pbr_uv_tan.spv"),
                    frag: PathBuf::from("fs_pbr_bc_mr_nm_tex.spv"),
                    vertex_format,
                }
            } else if has_bc && !has_nm && !has_mr && !has_vertex_colors {
                GraphicsPipelineDescriptor {
                    vert: PathBuf::from("vs_pbr_uv.spv"),
                    frag: PathBuf::from("fs_pbr_bc_tex.spv"),
                    vertex_format,
                }
            } else if !has_nm && !has_bc && !has_mr && !has_vertex_colors {
                GraphicsPipelineDescriptor {
                    vert: PathBuf::from("vs_pbr_base.spv"),
                    frag: PathBuf::from("fs_pbr_base.spv"),
                    vertex_format,
                }
            } else {
                unimplemented!("Support more shad   er variants!")
            };

            renderer
                .create_resource(desc)
                .expect("Failed to create pipeline")
        }
        m => todo!("No support for this material yet {:?}", m),
    }
}
