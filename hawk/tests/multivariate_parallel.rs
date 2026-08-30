//! The parallel and sequential multivariate paths must agree **bitwise**
//! (M2 Part B step 14).
//!
//! Only compiled with `--features rayon`.
//!
//! # Why bitwise, and what makes it achievable
//!
//! Parallelism must not change the answer. The usual reason it does is that a
//! reduction is split differently across threads, so the summation order changes. The
//! fix for that is a deterministic accumulation order, not a looser test.
//!
//! Here the log term is accumulated into one slot per component and combined in index
//! order, in **both** paths. Nothing is summed across components inside the loop, so
//! there is no reduction whose split could vary, and the arithmetic is fixed
//! regardless of scheduling.
//!
//! # Sabotage
//!
//! Combining the per-component parts in reverse index order in the parallel path — a
//! change that is mathematically invisible and would pass any tolerance — turned
//! `agree_over_random_shapes` red, by one ulp:
//! `-89.82569657461426` against `-89.82569657461427`.
//!
//! The two fixed-shape tests stayed green, which is the useful part of the result: the
//! hand-chosen cases happen to round the same either way, and only the randomized
//! sweep found a shape where they do not. That is the same lesson as CLAUDE.md §3's
//! rule about fixed-seed cases, in a different guise. Recorded in
//! `docs/verification-log.md`.

#![cfg(feature = "rayon")]

mod common;

use common::Lcg;
use hawk::multivariate::{
    Observation, Parameters, negative_log_likelihood, negative_log_likelihood_parallel,
};
use proptest::prelude::*;

fn assert_identical(parameters: &Parameters, observation: &Observation, context: &str) {
    let sequential = negative_log_likelihood(parameters, observation).unwrap();
    let parallel = negative_log_likelihood_parallel(parameters, observation).unwrap();
    assert_eq!(
        sequential.to_bits(),
        parallel.to_bits(),
        "{context}: sequential {sequential:?} (bits {:#018x}) vs parallel \
         {parallel:?} (bits {:#018x}); difference {:e}",
        sequential.to_bits(),
        parallel.to_bits(),
        (sequential - parallel).abs()
    );
}

#[test]
fn agree_on_degenerate_and_tied_input() {
    let p = Parameters::new(vec![0.7, 0.5, 0.4], vec![0.1; 9], 1.3).unwrap();
    for events in [
        vec![vec![], vec![], vec![]],
        vec![vec![2.5], vec![], vec![]],
        vec![vec![1.0, 2.0, 2.0], vec![2.0, 3.0], vec![2.0]],
        vec![vec![0.0, 0.0], vec![0.0], vec![0.0, 5.0, 5.0]],
        vec![vec![2.0; 6], vec![2.0; 3], vec![2.0]],
    ] {
        let observation = Observation::new(&events, 6.0).unwrap();
        assert_identical(&p, &observation, &format!("{events:?}"));
    }
}

/// Repeated runs, because a scheduling-dependent bug is not deterministic: it can
/// agree on one run and differ on the next.
#[test]
fn agree_across_repeated_runs() {
    let d = 8;
    let mut rng = Lcg::new(0x5DEE_CE66_D1CE_4321);
    let horizon = 400.0;
    let events: Vec<Vec<f64>> = (0..d)
        .map(|_| {
            let mut times: Vec<f64> = (0..300).map(|_| rng.next_f64() * horizon).collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times
        })
        .collect();
    let baseline: Vec<f64> = (0..d).map(|i| 0.2 + 0.05 * i as f64).collect();
    let excitation: Vec<f64> = (0..d * d)
        .map(|k| 0.02 + 0.3 * (k % 5) as f64 / d as f64)
        .collect();
    let p = Parameters::new(baseline, excitation, 1.1).unwrap();
    let observation = Observation::new(&events, horizon).unwrap();

    let sequential = negative_log_likelihood(&p, &observation).unwrap();
    for run in 0..25 {
        let parallel = negative_log_likelihood_parallel(&p, &observation).unwrap();
        assert_eq!(
            sequential.to_bits(),
            parallel.to_bits(),
            "run {run}: parallel {parallel:?} vs sequential {sequential:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    #[test]
    fn agree_over_random_shapes(
        d in 1usize..=8,
        seed in 0u64..100_000,
        horizon in 1.0f64..60.0,
        tie_grid in prop::bool::ANY,
    ) {
        let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let events: Vec<Vec<f64>> = (0..d)
            .map(|_| {
                let n = rng.next_usize(30);
                let mut times: Vec<f64> = (0..n)
                    .map(|_| if tie_grid {
                        (rng.next_f64() * 5.0).floor() * horizon / 5.0
                    } else {
                        rng.next_f64() * horizon
                    })
                    .collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times
            })
            .collect();
        let baseline: Vec<f64> = (0..d).map(|_| 0.05 + rng.next_f64() * 2.0).collect();
        let excitation: Vec<f64> = (0..d * d).map(|_| rng.next_f64() * 0.8 / d as f64).collect();
        let p = Parameters::new(baseline, excitation, 0.1 + rng.next_f64() * 3.0).unwrap();
        let observation = Observation::new(&events, horizon).unwrap();

        let sequential = negative_log_likelihood(&p, &observation).unwrap();
        let parallel = negative_log_likelihood_parallel(&p, &observation).unwrap();
        prop_assert_eq!(sequential.to_bits(), parallel.to_bits(),
            "d={} sequential {:?} vs parallel {:?}", d, sequential, parallel);
    }
}
