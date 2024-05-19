use crate::backend::instance::Instance;
use crate::util::lifetime::LifetimeToken;

use ash::Entry;

use ash::extensions::ext;
use ash::vk;

use thiserror::Error;

use std::ffi::{CStr, CString};
use std::fmt::Write;
use std::os::raw::c_char;

#[allow(dead_code)]
const DISABLE_VALIDATION_LAYERS_ENV_VAR: &str = "TREKANTEN_DISABLE_VALIDATION_LAYERS";

fn validation_layers() -> Vec<CString> {
    vec![CString::new("VK_LAYER_KHRONOS_validation").expect("Failed to create CString")]
}

pub fn use_vk_validation() -> bool {
    #[cfg(feature = "validation-layers")]
    {
        std::env::var(DISABLE_VALIDATION_LAYERS_ENV_VAR).is_err()
    }

    #[cfg(not(feature = "validation-layers"))]
    {
        false
    }
}

pub fn choose_validation_layers(entry: &Entry) -> Vec<CString> {
    if use_vk_validation() {
        let requested = validation_layers();
        log::trace!("Requested vk layers:");
        crate::util::ffi::log_cstrings(&requested);

        let layers = match entry.enumerate_instance_layer_properties() {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };

        if layers.is_empty() {
            log::warn!("Layers requested but found no layers");
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
        crate::util::ffi::log_cstrings(&requested);
        requested
    } else {
        Vec::new()
    }
}

#[derive(Debug, Error)]
pub enum DebugUtilsError {
    #[error("Failed to create vulkan debug utils extension {0}")]
    Creation(vk::Result),
}

pub struct DebugUtilsEnabled {
    loader: ext::DebugUtils,
    callback_handle: vk::DebugUtilsMessengerEXT,
    _parent_lifetime_token: LifetimeToken<Instance>,
}

#[allow(dead_code)]
impl DebugUtilsEnabled {
    fn new(instance: &Instance) -> Result<Self, DebugUtilsError> {
        let loader = ext::DebugUtils::new(instance.entry(), instance.vk_instance());

        let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
            )
            .pfn_user_callback(Some(vk_debug_callback));

        let callback_handle = unsafe {
            loader
                .create_debug_utils_messenger(&info, None)
                .map_err(DebugUtilsError::Creation)?
        };

        Ok(Self {
            loader,
            callback_handle,
            _parent_lifetime_token: instance.lifetime_token(),
        })
    }
}

impl Drop for DebugUtilsEnabled {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.callback_handle, None);
        }
    }
}

pub struct DebugUtilsDisabled {}

#[allow(dead_code)]
pub enum DebugUtils {
    Enabled(DebugUtilsEnabled),
    Disabled(DebugUtilsDisabled),
}

impl DebugUtils {
    #[allow(unused)]
    pub fn new(instance: &Instance) -> Result<Self, DebugUtilsError> {
        #[cfg(not(feature = "validation-layers"))]
        {
            Ok(Self::Disabled(DebugUtilsDisabled {}))
        }
        #[cfg(feature = "validation-layers")]
        {
            if use_vk_validation() {
                Ok(Self::Enabled(DebugUtilsEnabled::new(instance)?))
            } else {
                Ok(Self::Disabled(DebugUtilsDisabled {}))
            }
        }
    }
}

#[allow(dead_code)]
unsafe fn write_maybe_null(mut s: &mut String, p: *const c_char) {
    if p.is_null() {
        write!(&mut s, "(NULL)").expect("vk_debug_callback failed to write");
    } else {
        write!(&mut s, "({:?})", CStr::from_ptr(p)).expect("vk_debug_callback failed to write");
    }
}

#[allow(dead_code)]
unsafe extern "system" fn vk_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;

    let callback_data = *p_callback_data;

    let mut message = String::new();

    write!(&mut message, "[{:?}]", message_type).expect("vk_debug_callback failed to write");
    write!(
        &mut message,
        "[ID {} ",
        callback_data.message_id_number as u32
    )
    .expect("vk_debug_callback failed to write");

    write_maybe_null(&mut message, callback_data.p_message_id_name);
    writeln!(&mut message, "]").expect("vk_debug_callback failed to write");

    write_maybe_null(&mut message, callback_data.p_message);

    if message_severity.contains(Severity::VERBOSE) {
        log::trace!("{}", message);
    }

    if message_severity.contains(Severity::INFO) {
        log::info!("{}", message);
    }

    if message_severity.contains(Severity::WARNING) {
        log::warn!("{}", message);
    }

    if message_severity.contains(Severity::ERROR) {
        log::error!("{}", message);
        // panic!("Got error from vulkan validation layers");
    }

    // According to the lunarg tutorial for the callback, false => don't bail out
    0
}
