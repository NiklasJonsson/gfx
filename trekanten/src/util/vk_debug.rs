use ash::extensions::ext;
use ash::vk;

use std::ffi::CStr;
use std::fmt::Write;
use std::os::raw::c_char;

use crate::instance::Instance;
use crate::util::lifetime::LifetimeToken;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DebugUtilsError {
    #[error("Failed to create vulkan debug utils extension {0}")]
    Creation(vk::Result),
}

pub struct DebugUtils {
    loader: ext::DebugUtils,
    callback_handle: vk::DebugUtilsMessengerEXT,
    _parent_lifetime_token: LifetimeToken<Instance>,
}

impl Drop for DebugUtils {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.callback_handle, None);
        }
    }
}

impl DebugUtils {
    pub fn new(instance: &Instance) -> Result<Self, DebugUtilsError> {
        let loader = ext::DebugUtils::new(instance.entry(), instance.vk_instance());

        let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
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

unsafe fn write_maybe_null(mut s: &mut String, p: *const c_char) {
    if p.is_null() {
        write!(&mut s, "(NULL)").expect("vk_debug_callback failed to write");
    } else {
        write!(&mut s, "({:?})", CStr::from_ptr(p)).expect("vk_debug_callback failed to write");
    }
}

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
    }

    // According to the lunarg tutorial for the callback, false => don't bail out
    0
}
