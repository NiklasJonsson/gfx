pub fn stride(elem_size: u16, elem_align: u16) -> u16 {
    let padding = if elem_size == elem_align { 0 } else { 1 };
    ((elem_size / elem_align) + padding) * elem_align
}
