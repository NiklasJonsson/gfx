use std::marker::PhantomData;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct ID {
    index: usize,
}

// Can't derive things on Handle because of PhantomData + generic
// https://github.com/rust-lang/rust/issues/26925
#[derive(Debug)]
pub struct Handle<T> {
    id: ID,
    ty: PhantomData<T>,
}
impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self {
            id: ID::default(),
            ty: PhantomData {},
        }
    }
}
impl<T> std::cmp::PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> std::cmp::Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> Handle<T> {
    fn new(id: ID) -> Self {
        Handle::<T> {
            id,
            ty: PhantomData {},
        }
    }

    pub fn index(&self) -> usize {
        self.id.index
    }

    pub fn as_buffered(&self) -> Handle<[T; 2]> {
        Handle::<[T; 2]>::new(self.id)
    }

    pub fn id(&self) -> ID {
        self.id
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle::<T>::new(self.id)
    }
}
impl<T> Copy for Handle<T> {}

// For buffered storage
// TODO: Make generic over array length
impl<T> Handle<[T; 2]> {
    pub fn as_unbuffered(self) -> Handle<T> {
        Handle::<T>::new(self.id)
    }
}

// Based on sparse sets:
// https://programmingpraxis.com/2012/03/09/sparse-sets/
// https://bitsquid.blogspot.com/2011/09/managing-decoupling-part-4-id-lookup.html
// https://blog.molecular-matters.com/2013/07/24/adventures-in-data-oriented-design-part-3c-external-references/

// TODO: Implement ID index reuse with generations, as sparse will grow bigger and bigger as it is
// now
pub struct Storage<T> {
    data: Vec<T>,
    dense: Vec<ID>,
    sparse: Vec<usize>,
}

const INVALID_DENSE_IDX: usize = usize::MAX;

impl<T> Storage<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add(&mut self, a: T) -> Handle<T> {
        assert_eq!(self.data.len(), self.dense.len());

        let sparse_idx = self.sparse.len();
        let dense_idx = self.dense.len();
        self.sparse.push(dense_idx);
        self.data.push(a);
        let id = ID { index: sparse_idx };

        self.dense.push(id);

        Handle::<T>::new(id)
    }

    pub fn remove(&mut self, h: Handle<T>) -> Option<T> {
        if !self.has(&h) {
            return None;
        }
        assert!(!self.sparse.is_empty());
        assert_eq!(self.data.len(), self.dense.len());

        let sparse_idx = h.index();
        let dense_idx = self.sparse[sparse_idx];

        if self.dense.len() > 1 {
            // Swap last and the one we want to remove
            let last = self.dense.len() - 1;
            self.dense.swap(dense_idx, last);
            self.data.swap(dense_idx, last);
            assert_eq!(self.sparse[self.dense[dense_idx].index], last);
            // Update the sparse -> dense idx
            self.sparse[self.dense[dense_idx].index] = dense_idx;
        }

        self.sparse[sparse_idx] = INVALID_DENSE_IDX;

        assert_eq!(*self.dense.last().unwrap(), h.id);

        self.dense.pop();
        Some(self.data.pop().unwrap())
    }

    pub fn has(&self, h: &Handle<T>) -> bool {
        if self.sparse.is_empty() || h.index() as usize >= self.sparse.len() {
            return false;
        }

        if self.sparse[h.index() as usize] == INVALID_DENSE_IDX {
            return false;
        }

        self.dense[self.sparse[h.index() as usize]] == h.id
    }

    pub fn get(&self, h: &Handle<T>) -> Option<&T> {
        if !self.has(h) {
            None
        } else {
            Some(&self.data[self.sparse[h.index()] as usize])
        }
    }

    pub fn get_mut(&mut self, h: &Handle<T>) -> Option<&mut T> {
        if !self.has(h) {
            None
        } else {
            Some(&mut self.data[self.sparse[h.index()] as usize])
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
}

impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            dense: Default::default(),
            sparse: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let mut m = Storage::default();
        let i0 = m.add(4);
        let i1 = m.add(10);
        let i2 = m.add(2000);

        assert_eq!(m.len(), 3);
        assert!(m.has(&i0));
        assert!(m.has(&i1));
        assert!(m.has(&i2));
        assert_eq!(*m.get(&i0).unwrap(), 4);
        assert_eq!(*m.get(&i1).unwrap(), 10);
        assert_eq!(*m.get(&i2).unwrap(), 2000);

        assert_eq!(m.get(&i0).copied(), m.get_mut(&i0).copied());
        assert_eq!(m.get(&i1).copied(), m.get_mut(&i1).copied());
        assert_eq!(m.get(&i2).copied(), m.get_mut(&i2).copied());
    }

    #[test]
    fn remove() {
        let mut m = Storage::new();
        let i0 = m.add(4);
        let r = m.remove(i0);
        assert_eq!(r.unwrap(), 4);
        assert!(!m.has(&i0));
        assert!(m.get(&i0).is_none());
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let i0 = m.add(5);
        assert_eq!(m.len(), 1);
        assert!(m.has(&i0));

        let i1 = m.add(15);
        assert_eq!(m.len(), 2);
        assert!(m.has(&i1));

        assert_eq!(m.remove(i1).unwrap(), 15);
        assert_eq!(m.len(), 1);
        assert!(m.has(&i0));
        assert!(!m.has(&i1));

        let i2 = m.add(25);
        assert_eq!(m.len(), 2);
        assert!(!m.has(&i1));
        assert!(m.has(&i0));
        assert!(m.has(&i2));
    }

    fn add_int_range(s: &mut Storage<u32>, start: u32, end: u32) -> Vec<Handle<u32>> {
        (start..end).map(|x| s.add(x)).collect::<Vec<_>>()
    }

    fn remove_with_cond(s: &mut Storage<u32>, handles: &[Handle<u32>], cond: fn(usize) -> bool) {
        for (i, h) in handles.iter().enumerate() {
            assert!(s.has(h));
            assert_eq!(*s.get(h).unwrap() as usize, i);

            if cond(i) {
                assert_eq!(s.remove(*h).unwrap() as usize, i);
            }
        }
    }

    fn check_with_cond(s: &Storage<u32>, handles: &[Handle<u32>], cond: fn(usize) -> bool) {
        for (i, h) in handles.iter().enumerate() {
            if cond(i) {
                assert!(!s.has(h));
                assert!(std::matches!(s.get(h), None));
            } else {
                assert!(s.has(h));
                assert_eq!(*s.get(h).unwrap() as usize, i);
            }
        }
    }

    #[test]
    fn remove_even() {
        let mut m = Storage::new();
        let r = add_int_range(&mut m, 0, 100);

        let f = |x| x % 2 == 0;
        remove_with_cond(&mut m, &r, f);
        check_with_cond(&mut m, &r, f);
    }

    #[test]
    fn remove_first_segment() {
        let mut m = Storage::new();
        let r = add_int_range(&mut m, 0, 102);

        let f = |x| x < 30;
        remove_with_cond(&mut m, &r, f);
        check_with_cond(&mut m, &r, f);
    }

    #[test]
    fn remove_last_segment() {
        let mut m = Storage::new();
        let r = add_int_range(&mut m, 0, 102);

        let f = |x| x > 60;
        remove_with_cond(&mut m, &r, f);
        check_with_cond(&mut m, &r, f);
    }

    #[test]
    fn remove_middle_segment() {
        let mut m = Storage::new();
        let r = add_int_range(&mut m, 0, 102);

        let f = |x| x < 60 && x > 30;
        remove_with_cond(&mut m, &r, f);
        check_with_cond(&mut m, &r, f);
    }

    #[test]
    fn remove_first_and_last() {
        let mut m = Storage::new();
        let r = add_int_range(&mut m, 0, 102);

        let f = |x| x > 60 || x < 30;
        remove_with_cond(&mut m, &r, f);
        check_with_cond(&mut m, &r, f);
    }

    #[test]
    fn batch_add_remove() {
        let mut m = Storage::new();
        let r = add_int_range(&mut m, 0, 102);

        let f = |x| x > 60 || x < 30;
        remove_with_cond(&mut m, &r, f);
        check_with_cond(&m, &r, f);

        let r1 = add_int_range(&mut m, 0, 10);
        check_with_cond(&m, &r1, |_| false);
        let r2 = add_int_range(&mut m, 0, 13000);
        check_with_cond(&m, &r2, |_| false);
        let r3 = add_int_range(&mut m, 0, 1);
        check_with_cond(&m, &r3, |_| false);

        for h in &r2 {
            m.remove(*h);
        }

        check_with_cond(&m, &r1, |_| false);
        check_with_cond(&m, &r2, |_| true);
        check_with_cond(&m, &r3, |_| false);
    }
}
