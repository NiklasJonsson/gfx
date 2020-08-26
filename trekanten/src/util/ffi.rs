use std::ffi::CString;
use std::os::raw::c_char;

pub fn log_cstrings(a: &[CString]) {
    for cs in a {
        log::trace!("{:?}", cs);
    }
}

/// This will leak memory if vec_ptrs_to_cstring is not called
pub fn vec_cstring_to_raw(v: Vec<CString>) -> Vec<*const c_char> {
    v.into_iter()
        .map(|x| x.into_raw() as *const c_char)
        .collect::<Vec<_>>()
}

/// Call this to reclaim memory of the vec of c_chars
pub fn vec_cstring_from_raw(v: Vec<*const c_char>) -> Vec<CString> {
    v.iter()
        .map(|x| unsafe { CString::from_raw(*x as *mut c_char) })
        .collect::<Vec<_>>()
}
