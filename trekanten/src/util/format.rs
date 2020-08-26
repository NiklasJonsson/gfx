use ash::vk;

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Format(vk::Format);
// TODO: Define enum values so that size can be masked.
impl Format {
    pub fn size(&self) -> u32 {
        match *self {
            Self::FLOAT4 => 16,
            Self::FLOAT3 => 12,
            Self::FLOAT2 => 8,
            Self::FLOAT1 => 4,
            Self::RGBA_SRGB => 4,
            Self::RGBA_UNORM => 4,
            _ => unimplemented!("Missing case in match"),
        }
    }

    pub const FLOAT4: Self = Self(vk::Format::R32G32B32A32_SFLOAT);
    pub const FLOAT3: Self = Self(vk::Format::R32G32B32_SFLOAT);
    pub const FLOAT2: Self = Self(vk::Format::R32G32_SFLOAT);
    pub const FLOAT1: Self = Self(vk::Format::R32_SFLOAT);

    pub const RGBA_SRGB: Self = Self(vk::Format::R8G8B8A8_SRGB);
    pub const RGBA_UNORM: Self = Self(vk::Format::R8G8B8A8_UNORM);
}

impl From<Format> for vk::Format {
    fn from(f: Format) -> vk::Format {
        f.0
    }
}

impl From<vk::Format> for Format {
    fn from(f: vk::Format) -> Self {
        Self(f)
    }
}
