/*
https://registry.khronos.org/OpenGL/specs/gl/glspec45.core.pdf#page=159
7.6.2.2 Standard Uniform Block Layout

When using the std140 storage layout, structures will be laid out in buffer
storage with their members stored in monotonically increasing order based on their
location in the declaration. A structure and each structure member have a base
offset and a base alignment, from which an aligned offset is computed by rounding
the base offset up to a multiple of the base alignment. The base offset of the first
member of a structure is taken from the aligned offset of the structure itself. The
base offset of all other structure members is derived by taking the offset of the
last basic machine unit consumed by the previous member and adding one. Each
structure member is stored in memory at its aligned offset. The members of a top-
level uniform block are laid out in buffer storage by treating the uniform block as
a structure with a base offset of zero.

1. If the member is a scalar consuming N basic machine units, the base align-
ment is N.
2. If the member is a two- or four-component vector with components consum-
ing N basic machine units, the base alignment is 2N or 4N, respectively.
3. If the member is a three-component vector with components consuming N
basic machine units, the base alignment is 4N.
4. If the member is an array of scalars or vectors, the base alignment and array
stride are set to match the base alignment of a single array element, according
to rules (1), (2), and (3), and rounded up to the base alignment of a vec4. The
array may have padding at the end; the base offset of the member following
the array is rounded up to the next multiple of the base alignment.
5. If the member is a column-major matrix with C columns and R rows, the
matrix is stored identically to an array of C column vectors with R compo-
nents each, according to rule (4).
6. If the member is an array of S column-major matrices with C columns and
R rows, the matrix is stored identically to a row of S × C column vectors
with R components each, according to rule (4).
7. If the member is a row-major matrix with C columns and R rows, the matrix
is stored identically to an array of R row vectors with C components each,
according to rule (4).
8. If the member is an array of S row-major matrices with C columns and R
rows, the matrix is stored identically to a row of S × R row vectors with C
components each, according to rule (4).
9. If the member is a structure, the base alignment of the structure is N , where
N is the largest base alignment value of any of its members, and rounded
up to the base alignment of a vec4. The individual members of this sub-
structure are then assigned offsets by applying this set of rules recursively,
where the base offset of the first member of the sub-structure is equal to the
aligned offset of the structure. The structure may have padding at the end;
the base offset of the member following the sub-structure is rounded up to
the next multiple of the base alignment of the structure.
10. If the member is an array of S structures, the S elements of the array are laid
out in order, according to rule (9).

Shader storage blocks (see section 7.8) also support the std140 layout qual-
ifier, as well as a std430 qualifier not supported for uniform blocks. When using
the std430 storage layout, shader storage blocks will be laid out in buffer storage
identically to uniform and shader storage blocks using the std140 layout, except
that the base alignment and stride of arrays of scalars and vectors in rule 4 and of
structures in rule 9 are not rounded up a multiple of the base alignment of a vec4
 */

// Breaking it down:
// * structures will be laid out in buffer storage with their members stored in monotonically
//   increasing order based on their location in the declaration
//   => No re-arranging of members.
// * A structure and each structure member have a base offset and a base alignment, from which
//   an aligned offset is computed by rounding the base offset up to a multiple of the base alignment
//   => Introduces three concepts, base offset, base alignment and aligned offset.
// * The base offset of the first member of a structure is taken from the aligned offset of the structure itself.
//   => No padding before first member.
// * The base offset of all other structure members is derived by taking the offset of the last
//   basic machine unit consumed by the previous member and adding one.
//   => Base offset starts directly after preceding member.
// * Each structure member is stored in memory at its aligned offset.
//   => There is potential padding if base offset is not a multiple of aligned offset (as it is rounded up).
// * The members of a top-level uniform block are laid out in buffer storage by treating the uniform
//   block as a structure with a base offset of zero.
//   => Start at 0 for the first member.

/// # Safety
///
/// Std140 according to the opengl spec. Use the trekant::Std140 to derive
pub unsafe trait Std140: Copy {
    const SIZE: usize; // TODO: fixed width
    const ALIGNMENT: usize; // TODO: Fixed width
}

pub trait Std140Struct: Std140 {}

macro_rules! impl_std140 {
    ($ty:ty, $size:expr, $align:expr) => {
        unsafe impl Std140 for $ty {
            const SIZE: usize = $size;
            const ALIGNMENT: usize = $align;
        }
    };
}

macro_rules! impl_std140_scalar {
    ($ty:ty) => {
        // rule 1
        impl_std140!($ty, 4, 4);
        // rule 2
        impl_std140!([$ty; 2], 8, 8);
        impl_std140!([$ty; 4], 16, 16);
        // rule 3
        impl_std140!([$ty; 3], 12, 16);

        // partial support for rule 5, 7. Skipping 2 and 3 elem vecs.
        impl_std140!([[$ty; 4]; 4], 64, 16);
        impl_std140!([$ty; 16], 64, 16);
    };
}

impl_std140_scalar!(f32);
impl_std140_scalar!(i32);
impl_std140_scalar!(u32);

// rule 4 not supported

// rule 6, 8
unsafe impl<const N: usize> Std140 for [[f32; 16]; N] {
    const ALIGNMENT: usize = 16;
    const SIZE: usize = 64 * N;
}

// For rule 9, use the Std140 proc macro to derive the type.

// rule 10
unsafe impl<T: Std140Struct, const N: usize> Std140 for [T; N] {
    const SIZE: usize = crate::util::round_to_multiple(T::SIZE * N, Self::ALIGNMENT);
    const ALIGNMENT: usize = crate::util::round_to_multiple(T::ALIGNMENT, 16);
}
