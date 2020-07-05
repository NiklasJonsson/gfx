use crate::asset;
use crate::asset::{cache::Cache, storage::Handle, storage::Storage, TextureDescriptor};
use crate::common::Format;

use vulkano::device::Queue;
use vulkano::format::Format as VkFormat;
use vulkano::image::{immutable::ImmutableImage, Dimensions, ImageViewAccess};
use vulkano::sampler::Sampler;
use vulkano::sync::GpuFuture;

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct Texture {
    pub image: image::RgbaImage,
    pub format: Format,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Texture")
            .field(
                "image",
                &format_args!(
                    "RgbaImage{{ w: {}, h: {}}}",
                    self.image.width(),
                    self.image.height()
                ),
            )
            .field("format", &self.format)
            .finish()
    }
}

#[derive(Clone, Debug, Default)]
struct Stats {
    pub hits: usize,
    pub misses: usize,
}

#[derive(Default)]
pub struct Textures {
    cache: Cache<TextureDescriptor, Texture>,
    storage: Storage<Texture>,
    stats: Stats,
}

impl Textures {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn load(&mut self, tex_desc: &TextureDescriptor) -> Handle<Texture> {
        let h = match self.cache.get(tex_desc) {
            Some(h) => {
                self.stats.hits += 1;
                h
            }
            None => {
                self.stats.misses += 1;
                let t = Texture {
                    image: asset::load_image(tex_desc),
                    format: tex_desc.format,
                };
                let h = self.storage.add(t);
                self.cache.add(tex_desc.clone(), h);
                h
            }
        };

        log::debug!(
            "Cache hits: {} / {}",
            self.stats.hits,
            self.stats.misses + self.stats.hits
        );

        h
    }

    pub fn get(&self, h: Handle<Texture>) -> Option<&Texture> {
        self.storage.get(h)
    }
}

// TODO: We are storing the sampler here as well. We should not.
#[derive(Clone)]
pub struct TextureAccess {
    pub image_buffer: Arc<dyn ImageViewAccess + Send + Sync>,
    pub sampler: Arc<Sampler>,
}

// TODO: Smart on-gpu allocation?
// TODO: Merge this and Textures?
#[derive(Default)]
pub struct GpuTextures {
    // TODO: Reuse asset cache here?
    cache: HashMap<Handle<Texture>, TextureAccess>,
}

impl GpuTextures {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn upload(
        &mut self,
        queue: &Arc<Queue>,
        textures: &Textures,
        handle: Handle<Texture>,
    ) -> TextureAccess {
        if let Some(access) = self.cache.get(&handle) {
            return access.clone();
        }

        let tex = textures
            .get(handle)
            .expect("Can't access texture for uploading!");
        let vk_format: VkFormat = tex.format.into();
        // TODO: Support mip maps
        let width = tex.image.width();
        let height = tex.image.height();
        let (buf, fut) = ImmutableImage::from_iter(
            tex.image.clone().into_raw().into_iter(),
            Dimensions::Dim2d { width, height },
            vk_format,
            Arc::clone(queue),
        )
        .expect("Unable to upload texture");

        // TODO: handle other sampling types
        // TODO: Move the sampler out of here
        let sampler = Sampler::simple_repeat_linear(Arc::clone(queue.device()));
        let tex_access = TextureAccess {
            image_buffer: buf,
            sampler,
        };

        // TODO: Make this async again.
        // Block the caller until we are done.
        fut.then_signal_fence_and_flush();

        tex_access
    }
}
