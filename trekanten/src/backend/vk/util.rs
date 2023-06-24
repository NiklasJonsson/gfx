/// Compute the stride requirement for elements with size 'elem_size' and alignment 'elem_align'.
///
/// Stride is used when laying out elements in a sequence in memory.
/// Assume a struct has size 9, and its member with the highest alignment requirement needs to be aligned to 8 bytes.
/// Then, to make sure we respect the alignment requirement, the stride needs to be 16.
///
/// In other words, the stride is the closest multiple of the alignment that is larger than the size.
///
pub fn stride(elem_size: u16, elem_align: u16) -> u16 {
    assert!(elem_size != 0 && elem_align != 0);

    let padding = if elem_size % elem_align == 0 { 0 } else { 1 };
    ((elem_size / elem_align) + padding) * elem_align
}

#[cfg(test)]
mod tests {
    use super::stride;

    #[test]
    fn test_stride() {
        assert_eq!(4, stride(4, 4));
        assert_eq!(16, stride(15, 8));
        assert_eq!(16, stride(9, 8));
        assert_eq!(8, stride(3, 8));
        assert_eq!(32, stride(3, 32));
        assert_eq!(32, stride(32, 8));
        assert_eq!(16, stride(16, 8));
    }
}
