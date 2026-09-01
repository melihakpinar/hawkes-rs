//! Measures the rayon path against the sequential one. Numbers only.
//!
//! `cargo run --release -p hawkes-benchmarks --features rayon --bin parallel_probe`;
//! the result is `benchmarks/results/multivariate-parallel.json`.
use std::hint::black_box;
use std::time::Instant;

use hawkes::multivariate::{
    Observation, Parameters, negative_log_likelihood, negative_log_likelihood_parallel, simulate,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    for &d in &[2usize, 5, 10, 20] {
        let baseline: Vec<f64> = (0..d).map(|i| 0.2 + 0.02 * i as f64).collect();
        let mut excitation = vec![0.0; d * d];
        for i in 0..d {
            excitation[i * d + i] += 0.05;
            excitation[i * d + (i + 1) % d] += 0.30;
        }
        let p = Parameters::new(baseline, excitation, 1.0).unwrap();
        let horizon = 200_000.0 / d as f64;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let events = simulate(&p, horizon, &mut rng).unwrap();
        let observation = Observation::new(&events, horizon).unwrap();
        let n: usize = events.iter().map(Vec::len).sum();

        let time = |f: &dyn Fn() -> f64| {
            black_box(f());
            let mut samples = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                black_box(f());
                samples.push(start.elapsed().as_secs_f64());
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            samples[2]
        };
        let sequential = time(&|| negative_log_likelihood(&p, &observation).unwrap());
        let parallel = time(&|| negative_log_likelihood_parallel(&p, &observation).unwrap());
        println!(
            "d={d:<3} n={n:<8} sequential={sequential:.6}s  parallel={parallel:.6}s  \
             ratio={:.2}x",
            parallel / sequential
        );
    }
}
