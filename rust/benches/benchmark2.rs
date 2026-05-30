use criterion::{criterion_group, criterion_main, Criterion};

fn bench_zeros(c: &mut Criterion) {
    let block_size = 64;
    c.bench_function("vec_macro", |b| {
        b.iter(|| vec![0.0; block_size])
    });

    let zero_buffer = vec![0.0; block_size];
    c.bench_function("clone_buffer", |b| {
        b.iter(|| zero_buffer.clone())
    });
}

criterion_group!(benches, bench_zeros);
criterion_main!(benches);
