use std::env::current_dir;
use std::iter::repeat_with;
use std::sync::Arc;

use arrow_array::Float32Array;
use criterion::{Criterion, criterion_group, criterion_main};
use lance::dataset::Dataset;
use lance::index::ann::{FlatIndex, SearchParams};
use rand::Rng;

pub fn generate_random_array(n: usize) -> Arc<Float32Array> {
    let mut rng = rand::thread_rng();
    Arc::new(Float32Array::from(
        repeat_with(|| rng.gen::<f32>())
            .take(n)
            .collect::<Vec<f32>>(),
    ))
}

fn bench_search(c: &mut Criterion) {
    const NUM_THREADS: usize = 8;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(NUM_THREADS)
        .build()
        .unwrap();

    c.bench_function("vec-flat-index(1024 / 1M)", move |b| {
        b.to_async(&runtime).iter(|| async {
            let dataset_uri = current_dir().unwrap().join("vec_data");
            let dataset = Dataset::open(dataset_uri.as_path().to_str().unwrap())
                .await
                .unwrap();

            let index = FlatIndex::new(&dataset, "vec".to_string());
            let params = SearchParams {
                key: generate_random_array(1024),
                k: 10,
                nprob: 0,
            };
            index.search(&params).await.unwrap();
        })
    });
}

criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_search);
criterion_main!(benches);
