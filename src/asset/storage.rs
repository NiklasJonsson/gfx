use std::marker::PhantomData;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct ID {
    index: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Handle<A> {
    id: ID,
    ty: PhantomData<A>,
}

impl<A> Handle<A> {
    fn new(id: ID) -> Self {
        Handle::<A> {
            id,
            ty: PhantomData{}
        }
    }
}

// Can't derive clone because of the phantom-data member + no bounds on A
// https://github.com/rust-lang/rust/issues/26925
impl<A> Clone for Handle<A> {
    fn clone(&self) -> Self {
        Handle::<A>::new(self.id)
    }
}
impl<A> Copy for Handle<A> {}

// Based on sparse sets:
// https://programmingpraxis.com/2012/03/09/sparse-sets/
// https://bitsquid.blogspot.com/2011/09/managing-decoupling-part-4-id-lookup.html
// https://blog.molecular-matters.com/2013/07/24/adventures-in-data-oriented-design-part-3c-external-references/

// TODO: Implement ID index reuse with generations, as sparse will grow bigger and bigger as it is
// now
pub struct Storage<A> {
    data: Vec<A>,
    dense: Vec<ID>,
    sparse: Vec<usize>,
}

const INVALID_DENSE_IDX: usize = usize::MAX;

impl<A> Storage<A> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add(&mut self, a: A) -> Handle<A> {
        assert_eq!(self.data.len(), self.dense.len());
        //assert!(self.free.iter().any() || self.sparse.len() < u32::MAX as usize, "Ran out of ids!");
 
        let sparse_idx = self.sparse.len();
        let dense_idx = self.dense.len();
        self.sparse.push(dense_idx);
        self.data.push(a);
        let id = ID {
            index: sparse_idx,
        };

        self.dense.push(id);

        Handle::<A>::new(id)
    }

    pub fn remove(&mut self, h: Handle<A>) -> Option<A> {
        if !self.has(h) {
            return None;
        }
        assert!(!self.sparse.is_empty());
        assert_eq!(self.data.len(), self.dense.len());

        let sparse_idx = h.id.index;
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
        return Some(self.data.pop().unwrap());
    }

    pub fn has(&self, h: Handle<A>) -> bool {
        if self.sparse.is_empty() || h.id.index as usize >= self.sparse.len() {
            return false;
        }

        if self.sparse[h.id.index as usize] == INVALID_DENSE_IDX {
            return false;
        }

        self.dense[self.sparse[h.id.index as usize]] == h.id
    }

    pub fn get(&self, h: Handle<A>) -> Option<&A> {
        if !self.has(h) {
            None
        } else {
            Some(&self.data[self.sparse[h.id.index] as usize])
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl<A> Default for Storage<A> {
    fn default() -> Self {
        Storage::<A> {
            data: Vec::new(),
            dense: Vec::new(),
            sparse: Vec::new(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let mut m = Storage::new();
        let i0 = m.add(4);
        let i1 = m.add(10);
        let i2 = m.add(2000);

        assert_eq!(m.size(), 3);
        assert!(m.has(i0));
        assert!(m.has(i1));
        assert!(m.has(i2));
        assert_eq!(*m.get(i0).unwrap(), 4);
        assert_eq!(*m.get(i1).unwrap(), 10);
        assert_eq!(*m.get(i2).unwrap(), 2000);
    }

    #[test]
    fn remove() {
        let mut m = Storage::new();
        let i0 = m.add(4);
        let r = m.remove(i0);
        assert_eq!(r.unwrap(), 4);
        assert!(!m.has(i0));
        assert!(m.get(i0).is_none());
        assert_eq!(m.size(), 0);
        assert!(m.is_empty());

        let i0 = m.add(5);
        assert_eq!(m.size(), 1);
        assert!(m.has(i0));

        let i1 = m.add(15);
        assert_eq!(m.size(), 2);
        assert!(m.has(i1));

        assert_eq!(m.remove(i1).unwrap(), 15);
        assert_eq!(m.size(), 1);
        assert!(m.has(i0));
        assert!(!m.has(i1));

        let i2 = m.add(25);
        assert_eq!(m.size(), 2);
        assert!(!m.has(i1));
        assert!(m.has(i0));
        assert!(m.has(i2));
    }

    fn add_int_range(s: &mut Storage<u32>, start: u32, end: u32) -> Vec<Handle<u32>> {

        (start..end).map(|x| s.add(x)).collect::<Vec::<_>>()
    }


    fn remove_with_cond(s: &mut Storage<u32>, handles: &[Handle<u32>], cond: fn(usize) -> bool) {
        for (i, h) in handles.iter().enumerate() {
            assert!(s.has(*h));
            assert_eq!(*s.get(*h).unwrap() as usize, i);
             
            if cond(i) {
                assert_eq!(s.remove(*h).unwrap() as usize, i);
            }
        }

    }

    fn check_with_cond(s: &Storage<u32>, handles: &[Handle<u32>], cond: fn(usize) -> bool) {
        for (i, h) in handles.iter().enumerate() {
            if cond(i) {
                assert!(!s.has(*h));
                assert!(std::matches!(s.get(*h), None));
            } else {
                assert!(s.has(*h));
                assert_eq!(*s.get(*h).unwrap() as usize, i);
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

