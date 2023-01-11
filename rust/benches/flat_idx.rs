use arrow_array::Float32Array;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use lance::dataset::Dataset;
use lance::index::ann::{FlatIndex, SearchParams};
use rand;
use rand::Rng;
use std::iter::repeat_with;
use std::sync::Arc;
// use criterion::async_executor::tokio::runtime::Runtime;

pub fn generate_random_array(n: usize) -> Arc<Float32Array> {
    let mut rng = rand::thread_rng();
    Arc::new(Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    ))
}

fn bench_search(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(8).build().unwrap();
    c.bench_function("iter", move |b| {
        b.to_async(&runtime).iter(|| async {
            let dataset = Dataset::open("/home/lei/work/lance/rust/vec_data")
                .await
                .unwrap();
            println!("Dataset schema: {:?}", dataset.schema());

            let index = FlatIndex::new(&dataset, "vec".to_string());
            let params = SearchParams {
                key: generate_random_array(1024),
                k: 10,
                nprob: 0,
            };
            let scores = index.search(&params).await.unwrap();
            println!("scores: {:?}\n", scores);
        })
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_search);
criterion_main!(benches);
