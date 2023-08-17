use ash::Entry;

use ash::vk;

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

fn has_extension(available: &[ash::vk::ExtensionProperties], req: &CStr) -> bool {
    for avail in available.iter() {
        let a = unsafe { CStr::from_ptr(avail.extension_name.as_ptr()) };
        if a == req {
            return true;
        }
    }
    false
}

fn check_extensions(
    required: &[&CStr],
    available: &[ash::vk::ExtensionProperties],
) -> Result<(), InstanceError> {
    for &req in required.iter() {
        if !has_extension(available, req) {
            let string = req
                .to_owned()
                .into_string()
                .expect("CString to String conversion failed");
            return Err(InstanceError::MissingExtension(string));
        }
    }

    Ok(())
}

fn choose_instance_extensions<W>(
    entry: &Entry,
    window: &W,
) -> Result<Vec<*const c_char>, InstanceError>
where
    W: raw_window_handle::HasRawWindowHandle + raw_window_handle::HasRawDisplayHandle,
{
    let available = entry
        .enumerate_instance_extension_properties(None)
        .map_err(|e| InstanceError::InternalVulkan(e, "Instance extension enumeration"))?;

    let mut required = Vec::new();

    for ext in
        ash_window::enumerate_required_extensions(window.raw_display_handle()).map_err(|e| {
            InstanceError::InternalVulkan(
                e,
                "Couldn't infer required window/surface extensions from window handle",
            )
        })?
    {
        assert!(!ext.is_null());
        required.push(unsafe { CStr::from_ptr(*ext) });
    }

    if super::validation_layers::use_vk_validation() {
        if !has_extension(&available, ash::extensions::ext::DebugUtils::name()) {
            log::warn!("Tried to enable debug utils extension, for validation layers but it is not supported");
        } else {
            required.push(ash::extensions::ext::DebugUtils::name());
        }
    }

    check_extensions(&required, &available)?;

    log::trace!("Choosing instance extensions: ");
    log_cstrings(&required);

    Ok(required.iter().map(|x| x.as_ptr()).collect())
}

impl Instance {
    pub fn new<W>(window: &W) -> Result<Self, InstanceError>
    where
        W: raw_window_handle::HasRawDisplayHandle + raw_window_handle::HasRawWindowHandle,
    {
        let entry = unsafe { Entry::load() }.expect("Failed to create Entry!");

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 2, 0),
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
                .map_err(InstanceError::Creation)?
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
