use ash::version::InstanceV1_0; // For destroy_instance
use ash::{version::EntryV1_0, vk, Entry};
use std::ffi::CStr;

use crate::util::ffi::{c_char, log_cstrings, vec_cstring_from_raw, vec_cstring_to_raw};
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

fn choose_instance_extensions<W: raw_window_handle::HasRawWindowHandle>(
    entry: &Entry,
    window: &W,
) -> Result<Vec<*const c_char>, InstanceError> {
    let available = entry
        .enumerate_instance_extension_properties()
        .map_err(|e| InstanceError::InternalVulkan(e, "Instance extension enumeration"))?;
    let mut required = ash_window::enumerate_required_extensions(window).map_err(|e| {
        InstanceError::InternalVulkan(
            e,
            "Couldn't infer required window/surface extensions from window handle",
        )
    })?;

    if super::validation_layers::use_vk_validation() {
        required.push(ash::extensions::ext::DebugUtils::name());
    }

    check_extensions(&required, &available)?;

    log::trace!("Choosing instance extensions: ");
    log_cstrings(&required);

    Ok(required.iter().map(|x| x.as_ptr()).collect())
}

impl Instance {
    pub fn new<W: raw_window_handle::HasRawWindowHandle>(
        window: &W,
    ) -> Result<Self, InstanceError> {
        let entry = Entry::new().expect("Failed to create Entry!");

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_version(1, 2, 0),
            ..Default::default()
        };

        let extensions_ptrs = choose_instance_extensions(&entry, window)?;

        let validation_layers = super::validation_layers::choose_validation_layers(&entry);
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
