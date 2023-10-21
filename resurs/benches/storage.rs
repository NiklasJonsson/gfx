use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use resurs::Storage;

const N_OPS: u32 = 10000;

pub fn insertion_10000(c: &mut Criterion) {
    c.bench_with_input(BenchmarkId::new("insertion", N_OPS), &N_OPS, |b, &size| {
        b.iter(|| {
            let mut storage: Storage<usize> = Storage::with_capacity(size);
            for _ in 0..size {
                storage.add(black_box(10));
            }
        })
    });
}

pub fn add_rm_10000(c: &mut Criterion) {
    c.bench_with_input(BenchmarkId::new("add_rm", N_OPS), &N_OPS, |b, &size| {
        b.iter(|| {
            let mut storage: Storage<usize> = Storage::with_capacity(size);
            let mut handles = Vec::with_capacity(size as usize);
            for _ in 0..size {
                let h = storage.add(black_box(10));
                handles.push(h);
            }

            for h in handles {
                storage.remove(h);
            }
        })
    });
}

criterion_group!(benches, insertion_10000, add_rm_10000);
criterion_main!(benches);
