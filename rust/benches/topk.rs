use std::iter::repeat_with;
use std::sync::Arc;
use arrow_array::{ArrayRef, Float32Array};
use arrow_ord::sort::sort_to_indices;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

use lance::index::ann::find_min_k;


pub fn generate_random_array(n: usize) -> Arc<Float32Array> {
    let mut rng = rand::thread_rng();
    Arc::new(Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    ))
}

fn bench_top_k(c: &mut Criterion) {
    c.bench_function("Top10From1M", |b| {
        let topk = 10;
        let card = 1000000;
        let mut arr: Vec<f32> = generate_random_array(card).values().to_vec();
        let mut indices: Vec<u64> = (0..card as u64).collect();
        b.iter(|| find_min_k(&mut arr, &mut indices, topk));
    });
    c.bench_function("Top10From1MSort", |b| {
        let topk = 10;
        let card = 1000000;
        let arr: ArrayRef = Arc::new(Float32Array::from(generate_random_array(card).values().to_vec()));
        b.iter(|| sort_to_indices(&arr, None, Some(topk)));
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_top_k);
criterion_main!(benches);

