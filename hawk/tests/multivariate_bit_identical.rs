//! The multivariate value-only and value+gradient paths must agree **bitwise**
//! (M2 Part B step 10).
//!
//! Same contract and same reasoning as `bit_identical_evaluation.rs` for the
//! univariate pair: the two loops share the arithmetic that decides the value —
//! `advance_excitation_state`, `compensator_contribution`, and the
//! `(alpha*beta)*state` grouping — and differ only in what they accumulate alongside
//! it. There is no numerical reason for them to differ, so the correct tolerance is
//! zero.
//!
//! `negative_log_likelihood` here has never delegated to the gradient path; it was
//! written as a separate loop from the start, having learned that from M1
//! (`docs/positioning-probe.md` part 3).
//!
//! # Sabotage
//!
//! Rewriting the value-only loop's `compensator_contribution` as `1 - exp(-x)` turned
//! `agree_on_events_packed_against_the_horizon` red and nothing else — the same
//! ulp-scale divergence that a tolerance would wave through. Dropping a component
//! from the value-only intensity sum turned every multi-component case red. Recorded
//! in `docs/verification-log.md`.

mod common;

use common::Lcg;
use hawk::multivariate::{
    Observation, Parameters, negative_log_likelihood, negative_log_likelihood_and_gradient,
};
use proptest::prelude::*;

fn assert_bit_identical(parameters: &Parameters, observation: &Observation, context: &str) {
    let value_only = negative_log_likelihood(parameters, observation).unwrap();
    let (with_gradient, _) = negative_log_likelihood_and_gradient(parameters, observation).unwrap();
    assert_eq!(
        value_only.to_bits(),
        with_gradient.to_bits(),
        "{context}: value-only {value_only:?} (bits {:#018x}) vs value+gradient \
         {with_gradient:?} (bits {:#018x}); difference {:e}. These must agree \
         exactly, not approximately.",
        value_only.to_bits(),
        with_gradient.to_bits(),
        (value_only - with_gradient).abs()
    );
}

fn parameters(baseline: Vec<f64>, excitation: Vec<f64>, decay: f64) -> Parameters {
    Parameters::new(baseline, excitation, decay).unwrap()
}

#[test]
fn agree_on_degenerate_input() {
    let p = parameters(vec![0.9, 0.4, 0.6], vec![0.1; 9], 1.7);
    for events in [
        vec![vec![], vec![], vec![]],
        vec![vec![2.5], vec![], vec![]],
        vec![vec![0.0], vec![5.0], vec![]],
        vec![vec![0.0, 5.0], vec![0.0, 5.0], vec![2.5]],
    ] {
        let observation = Observation::new(&events, 5.0).unwrap();
        assert_bit_identical(&p, &observation, &format!("{events:?}"));
    }
}

#[test]
fn agree_on_tied_input() {
    let p = parameters(vec![0.7, 0.5], vec![0.3, 0.2, 0.1, 0.25], 1.3);
    for events in [
        vec![vec![1.0, 2.0, 2.0, 3.0], vec![2.0, 4.0]],
        vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        vec![vec![2.0; 7], vec![2.0; 3]],
        vec![vec![1.0, 1.0, 1.0], vec![1.0, 5.0, 5.0]],
    ] {
        let observation = Observation::new(&events, 6.0).unwrap();
        assert_bit_identical(&p, &observation, &format!("{events:?}"));
    }
}

/// The case that a tolerance-based test cannot see; see
/// `bit_identical_evaluation.rs` for why it needs the sum itself to be tiny.
#[test]
fn agree_on_events_packed_against_the_horizon() {
    let horizon = 1.0;
    for events in [
        vec![
            vec![horizon - 1e-11, horizon - 1e-12],
            vec![horizon - 1e-13],
        ],
        vec![vec![horizon - 1e-13, horizon], vec![horizon - 1e-14]],
        vec![vec![horizon - 1e-9], vec![horizon - 1e-10, horizon]],
    ] {
        for (baseline, excitation, decay) in [
            (vec![0.5, 0.6], vec![0.3, 0.2, 0.1, 0.25], 1.0),
            (vec![2.0, 1.0], vec![0.45, 0.4, 0.3, 0.4], 5.0),
        ] {
            let p = parameters(baseline, excitation, decay);
            let observation = Observation::new(&events, horizon).unwrap();
            assert_bit_identical(&p, &observation, &format!("{events:?}"));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn agree_over_random_parameters(
        d in 1usize..=5,
        seed in 0u64..100_000,
        horizon in 1.0f64..80.0,
        tie_grid in prop::bool::ANY,
    ) {
        let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let events: Vec<Vec<f64>> = (0..d)
            .map(|_| {
                let n = rng.next_usize(40);
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
        let baseline: Vec<f64> = (0..d).map(|_| 0.01 + rng.next_f64() * 4.0).collect();
        let excitation: Vec<f64> = (0..d * d).map(|_| rng.next_f64() * 0.9 / d as f64).collect();
        let decay = 0.01 + rng.next_f64() * 6.0;

        let p = Parameters::new(baseline, excitation, decay).unwrap();
        let observation = Observation::new(&events, horizon).unwrap();
        let value_only = negative_log_likelihood(&p, &observation).unwrap();
        let (with_gradient, _) = negative_log_likelihood_and_gradient(&p, &observation).unwrap();
        prop_assert_eq!(value_only.to_bits(), with_gradient.to_bits(),
            "value-only {:?} vs value+gradient {:?}", value_only, with_gradient);
    }
}
