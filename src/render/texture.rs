use crate::common::Format;

#[derive(Clone)]
pub struct Texture {
    pub image: image::RgbaImage,
    pub coord_set: u32,
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
            .field("coord_set", &self.coord_set)
            .finish()
    }
}
