use ash::vk;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Format {
    vk_format: vk::Format,
}

impl From<Format> for vk::Format {
    fn from(f: Format) -> vk::Format {
        f.vk_format
    }
}

impl From<vk::Format> for Format {
    fn from(f: vk::Format) -> Self {
        Self { vk_format: f }
    }
}
