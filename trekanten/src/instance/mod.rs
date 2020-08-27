use ash::version::InstanceV1_0; // For destroy_instance
use ash::{version::EntryV1_0, vk, Entry};
use std::ffi::{CStr, CString};

use crate::util::ffi::{c_char, vec_cstring_from_raw, vec_cstring_to_raw, log_cstrings};
use crate::util::lifetime::LifetimeToken;

pub mod error;
pub use error::*;

pub struct Instance {
    entry: Entry,
    vk_instance: ash::Instance,
    lifetime_token: LifetimeToken<Self>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        if !self.lifetime_token.is_unique() {
            // TODO: Can we assert/panic here?
            log::error!("Instance destroyed but there are still children alive!");
        }
        unsafe {
            self.vk_instance.destroy_instance(None);
        }
    }
}

fn check_extensions<T: AsRef<CStr>>(
    required: &[T],
    available: &[ash::vk::ExtensionProperties],
) -> Result<(), InstanceError> {
    for req in required.iter() {
        let mut found = false;
        for avail in available.iter() {
            let a = unsafe { CStr::from_ptr(avail.extension_name.as_ptr()) };
            log::trace!("Available vk instance extension: {:?}", avail);
            if a == req.as_ref() {
                found = true;
            }
        }

        if !found {
            let string: String = req
                .as_ref()
                .to_owned()
                .into_string()
                .expect("CString to String failed");
            return Err(InstanceError::MissingExtension(string));
        }
    }

    Ok(())
}

const DISABLE_VALIDATION_LAYERS_ENV_VAR: &str = "TREK_DISABLE_VALIDATION_LAYERS";

fn validation_layers() -> Vec<CString> {
    vec![CString::new("VK_LAYER_KHRONOS_validation").expect("Failed to create CString")]
}

fn use_vk_validation() -> bool {
    std::env::var(DISABLE_VALIDATION_LAYERS_ENV_VAR).is_err()
}

pub fn choose_validation_layers(entry: &Entry) -> Vec<CString> {
    if use_vk_validation() {
        let requested = validation_layers();
        log::trace!("Requested vk layers:");
        log_cstrings(&requested);

        let layers = match entry.enumerate_instance_layer_properties() {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };

        if layers.is_empty() {
            log::trace!("Found no layers");
        }

        for req in requested.iter() {
            let mut found = false;
            for layer in layers.iter() {
                let l = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                log::trace!("Found vk layer: {:?}", layer);
                if l == req.as_c_str() {
                    found = true;
                }
            }

            if !found {
                return Vec::new();
            }
        }

        log::trace!("Choosing layers:");
        log_cstrings(&requested);
        requested
    } else {
        Vec::new()
    }
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
fn required_window_extensions() ->  Vec<&'static CStr> {
    vec![
        ash::extensions::khr::Surface::name(),
        ash::extensions::khr::XlibSurface::name(),
    ]
}

#[cfg(target_os = "macos")]
fn required_window_extensions() -> Vec<&'static CStr> {
    vec![
        ash::extensions::khr::Surface::name(),
        ash::extensions::mvk::MacOSSurface::name(),
    ]
}

#[cfg(all(windows))]
fn required_window_extensions() -> Vec<&'static CStr> {
    vec![
        ash::extensions::khr::Surface::name(),
        ash::extensions::khr::Win32Surface::name(),
    ]
}

fn choose_instance_extensions(
    entry: &Entry,
) -> Result<Vec<*const c_char>, InstanceError> {
    let available = entry
        .enumerate_instance_extension_properties()
        .map_err(|e| InstanceError::InternalVulkan(e, "Instance extension enumeration"))?;
    let mut required = required_window_extensions();

    if use_vk_validation() {
        required.push(ash::extensions::ext::DebugUtils::name());
    }

    check_extensions(&required, &available)?;

    log::trace!("Choosing instance extensions: ");
    log_cstrings(&required);

    Ok(required.iter().map(|x| x.as_ptr()).collect())
}

impl Instance {
    pub fn new() -> Result<Self, InstanceError> {
        let entry = Entry::new().expect("Failed to create Entry!");

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_version(1, 2, 0),
            ..Default::default()
        };

        let extensions_ptrs = choose_instance_extensions(&entry)?;

        let validation_layers = choose_validation_layers(&entry);
        let layers_ptrs = vec_cstring_to_raw(validation_layers);

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions_ptrs)
            .enabled_layer_names(&layers_ptrs);

        let vk_instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_err(InstanceError::from)?
        };

        let _owned_layers = vec_cstring_from_raw(layers_ptrs);

        let lifetime_token = LifetimeToken::<Instance>::new();

        let instance = Instance {
            entry,
            vk_instance,
            lifetime_token,
        };

        Ok(instance)
    }

    pub fn lifetime_token(&self) -> LifetimeToken<Self> {
        self.lifetime_token.clone()
    }

    pub fn vk_instance(&self) -> &ash::Instance {
        &self.vk_instance
    }

    pub fn vk_entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn entry(&self) -> &Entry {
        &self.entry
    }
}
