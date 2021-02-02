use super::Handle;

// Implementation is based on freelist array from https://ourmachinery.com/post/data-structures-part-1-bulk-data/
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct ID {
    index: u32,
    generation: u32,
}

impl std::fmt::Display for ID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ID {{ i: {}, g: {}}}", self.index, self.generation)
    }
}

enum ItemContent<T> {
    Data(T),
    NextFree(u32),
}

impl<T> ItemContent<T> {
    #[inline]
    pub fn as_ref(&self) -> ItemContent<&T> {
        match self {
            Self::Data(d) => ItemContent::Data(d),
            Self::NextFree(n) => ItemContent::NextFree(*n),
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> ItemContent<&mut T> {
        match self {
            Self::Data(d) => ItemContent::Data(d),
            Self::NextFree(n) => ItemContent::NextFree(*n),
        }
    }

    #[inline]
    pub fn data(self) -> Option<T> {
        match self {
            Self::Data(d) => Some(d),
            Self::NextFree(_) => None,
        }
    }
}

struct Item<T> {
    content: ItemContent<T>,
    generation: u32,
}

impl<T> Item<T> {
    #[inline]
    pub fn as_ref(&self) -> Item<&T> {
        Item {
            content: self.content.as_ref(),
            generation: self.generation,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Item<&mut T> {
        Item {
            content: self.content.as_mut(),
            generation: self.generation,
        }
    }

    #[inline]
    pub fn data(self) -> Option<T> {
        self.content.data()
    }
}

pub struct DrainFilter<'a, F, T>
where
    F: FnMut(&mut T) -> bool,
{
    storage: &'a mut Storage<T>,
    i: u32,
    pred: F,
}

impl<'a, F, T> DrainFilter<'a, F, T>
where
    F: FnMut(&mut T) -> bool,
{
    pub(crate) fn new(storage: &'a mut Storage<T>, pred: F) -> Self {
        Self {
            storage,
            i: 0,
            pred,
        }
    }
}

impl<'a, F, T> Iterator for DrainFilter<'a, F, T>
where
    F: FnMut(&mut T) -> bool,
{
    type Item = (Handle<T>, T);
    fn next(&mut self) -> Option<Self::Item> {
        while (self.i as usize) < self.storage.items.len() {
            let mut handle: Option<Handle<T>> = None;
            if let Item {
                content: ItemContent::Data(data),
                generation,
            } = &mut self.storage.items[self.i as usize]
            {
                if (self.pred)(data) {
                    handle = Some(Handle::<T>::new(ID {
                        index: self.i as u32,
                        generation: *generation,
                    }));
                }
            }
            if let Some(handle) = handle {
                let data = self.storage.remove(handle).expect("Just checked this");
                return Some((handle, data));
            }
            assert!(self.i < u32::MAX);
            self.i += 1;
        }

        return None;
    }
}

impl<'a, F, T> std::ops::Drop for DrainFilter<'a, F, T>
where
    F: FnMut(&mut T) -> bool,
{
    fn drop(&mut self) {
        while let Some(_) = self.next() {}
    }
}

pub struct Storage<T> {
    items: Vec<Item<T>>,
    next_free: u32,
    n_items: u32,
}

impl<T> Storage<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            next_free: 0,
            n_items: 0,
        }
    }

    #[inline]
    pub fn with_capacity(cap: u32) -> Self {
        Self {
            items: Vec::with_capacity(cap as usize),
            next_free: 0,
            n_items: 0,
        }
    }

    #[inline]
    pub fn add(&mut self, data: T) -> Handle<T> {
        let id = if self.next_free as usize == self.items.len() {
            debug_assert!(self.items.len() <= u32::MAX as usize);
            self.items.push(Item {
                content: ItemContent::Data(data),
                generation: 0,
            });
            self.next_free += 1;

            ID {
                index: (self.items.len() - 1) as u32,
                generation: 0,
            }
        } else {
            let index = self.next_free;
            let free = &mut self.items[index as usize];
            if let ItemContent::NextFree(next_free) = free.content {
                self.next_free = next_free;
            } else {
                panic!("Internal error: bad next_free value");
            }

            free.content = ItemContent::Data(data);
            ID {
                index,
                generation: free.generation,
            }
        };
        self.n_items += 1;

        Handle::<T>::new(id)
    }

    fn check_id(&self, id: ID) -> bool {
        let index = id.index as usize;

        if index >= self.items.len() {
            return false;
        }

        if id.generation != self.items[index].generation {
            return false;
        }

        return true;
    }

    #[inline]
    pub fn remove(&mut self, h: Handle<T>) -> Option<T> {
        let id = h.id();
        if !self.check_id(id) {
            return None;
        }

        let generation = id.generation.wrapping_add(1);
        let item = Item {
            content: ItemContent::NextFree(self.next_free),
            generation,
        };
        self.next_free = id.index;
        self.n_items -= 1;

        std::mem::replace(&mut self.items[id.index as usize], item).data()
    }

    #[inline]
    pub fn has(&self, h: &Handle<T>) -> bool {
        let id = h.id();
        if !self.check_id(id) {
            return false;
        }

        if let ItemContent::Data(_) = self.items[id.index as usize].content {
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn get(&self, h: &Handle<T>) -> Option<&T> {
        if !self.has(h) {
            None
        } else {
            self.items[h.id().index as usize].as_ref().data()
        }
    }

    #[inline]
    pub fn get_mut(&mut self, h: &Handle<T>) -> Option<&mut T> {
        if !self.has(h) {
            None
        } else {
            self.items[h.id().index as usize].as_mut().data()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_items == 0
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_items as usize
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter().filter_map(|x| x.as_ref().data())
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.items.iter_mut().filter_map(|x| x.as_mut().data())
    }

    #[inline]
    pub fn drain_filter<F>(&mut self, f: F) -> DrainFilter<'_, F, T>
    where
        F: FnMut(&mut T) -> bool,
    {
        DrainFilter::new(self, f)
    }
}

impl<T> Default for Storage<T> {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn drain_filter() {
        let mut m = Storage::default();
        let data = [10, 13, 16, 14, 1000, 2003];
        let mut handles = Vec::new();
        for d in data.iter() {
            handles.push(m.add(*d));
        }
        assert_eq!(m.len(), data.len());

        let mut expected = vec![10, 16, 14, 1000];
        for (_h, x) in m.drain_filter(|x| *x % 2 == 0) {
            let idx = expected.iter().position(|&e| e == x);
            assert!(idx.is_some());
            expected.remove(idx.unwrap());
        }
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn drain_filter_drop() {
        let mut m = Storage::default();
        let data = [10, 13, 16, 14, 1000, 2003];
        let mut handles = Vec::new();
        for d in data.iter() {
            handles.push(m.add(d));
        }

        {
            let _df = m.drain_filter(|x| *x % 2 != 0);
        }
        let expected = [10, 16, 14, 1000];
        assert_eq!(expected.len(), m.len());
        for d in m.iter() {
            assert!(expected.iter().find(|x| x == d).is_some());
        }
    }
}
