//! The two evaluation paths must agree **bitwise** (issue #13).
//!
//! `negative_log_likelihood` computes the value without the gradient;
//! `negative_log_likelihood_and_gradient` computes both. They share the arithmetic
//! that decides the value — `advance_excitation_state`, `intensity_at`,
//! `compensator_contribution` — but compose it in two separate loops, and nothing in
//! the type system stops those loops from drifting apart.
//!
//! # Why bitwise, and not a tolerance
//!
//! A tolerance would accept a change to the summation order, and summation order is
//! exactly what this is guarding. The two paths perform the same operations on the
//! same values in the same sequence; if that stops being true the results diverge in
//! the last bits first, long before any tolerance worth setting would notice. There
//! is no numerical reason for these two to differ at all, so the correct tolerance is
//! zero and the correct comparison is on the bit pattern.
//!
//! A red result here does not necessarily mean a wrong answer. It means the two paths
//! no longer agree exactly, and one of them must be brought back into line — or, if
//! the divergence is deliberate, this test and the doc comment on
//! `negative_log_likelihood` have to change together.
//!
//! # Sabotage
//!
//! Dropping `alpha` from the value-only path's final combination turned this red, as
//! did advancing the excitation state with `1.0` instead of the tie multiplicity —
//! the latter only on the tied cases, which is the correct blast radius.
//!
//! Replacing `compensator_contribution`'s `-exp_m1(-x)` with the algebraically equal
//! `1 - exp(-x)` did **not** turn it red on the first attempt. The per-event
//! difference is around `1e-16`, which sits below the ulp of the compensator sum and
//! disappears into it. `agree_on_events_packed_against_the_horizon` was added for
//! exactly that case and does catch it. Recorded in `docs/verification-log.md`.

use hawkes::univariate::{
    Observation, Parameters, negative_log_likelihood, negative_log_likelihood_and_gradient,
};
use proptest::prelude::*;

/// Compares the bit patterns, so `0.0` and `-0.0` would count as different. Neither
/// occurs here — the negative log-likelihood of a non-degenerate realization is not
/// signed zero — and treating them as different is the conservative direction anyway.
fn bits(value: f64) -> u64 {
    value.to_bits()
}

fn assert_bit_identical(parameters: &Parameters, observation: &Observation, context: &str) {
    let value_only = negative_log_likelihood(parameters, observation);
    let (with_gradient, _) = negative_log_likelihood_and_gradient(parameters, observation);
    assert_eq!(
        bits(value_only),
        bits(with_gradient),
        "{context}: value-only path gave {value_only:?} (bits {:#018x}), \
         value+gradient path gave {with_gradient:?} (bits {:#018x}); \
         difference {:e}. These must agree exactly, not approximately.",
        bits(value_only),
        bits(with_gradient),
        (value_only - with_gradient).abs(),
    );
}

#[test]
fn agree_on_degenerate_input() {
    let parameters = Parameters::new(0.9, 0.4, 1.7).unwrap();
    for (times, horizon) in [
        (vec![], 5.0),
        (vec![2.5], 5.0),
        (vec![0.0], 5.0),
        (vec![5.0], 5.0),
        (vec![0.0, 5.0], 5.0),
    ] {
        let observation = Observation::new(&times, horizon).unwrap();
        assert_bit_identical(&parameters, &observation, &format!("{times:?}"));
    }
}

#[test]
fn agree_on_tied_input() {
    // Ties take the branch that advances the state, so they exercise the shared
    // recursion helper from both loops.
    for (times, horizon) in [
        (vec![1.0, 2.0, 2.0, 3.0], 5.0),
        (vec![1.0, 2.0, 2.0, 2.0, 3.5], 6.0),
        (vec![0.0, 0.0, 1.5, 3.0, 5.0, 5.0], 5.0),
        (vec![2.0, 2.0, 2.0, 2.0], 4.0),
    ] {
        let parameters = Parameters::new(0.7, 0.5, 1.3).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();
        assert_bit_identical(&parameters, &observation, &format!("{times:?}"));
    }
}

/// Events packed against the horizon, where `1 - exp(-x)` and `-exp_m1(-x)` diverge.
///
/// This case exists because a sabotage found the suite without it. Rewriting the
/// value-only path's `-exp_m1(-x)` as the algebraically equal `1 - exp(-x)` left every
/// other test in this file green. The per-event difference is around `1e-16`, and
/// against a compensator sum of order 1 to 100 that is below the accumulator's own
/// ulp, so it vanishes into the total.
///
/// Making it visible needs the sum itself to be tiny: a handful of events within
/// `1e-11` of the horizon, so every contribution is around `1e-12` and a `1e-16`
/// perturbation is five orders of magnitude above the ulp of the sum rather than below
/// it. This is also precisely the regime `-exp_m1` was chosen for
/// (`univariate_loglikelihood.md` §5), so the case earns its place twice.
#[test]
fn agree_on_events_packed_against_the_horizon() {
    let horizon = 1.0;
    for (times, label) in [
        (
            vec![horizon - 1e-11, horizon - 1e-12, horizon - 1e-13],
            "1e-11 .. 1e-13",
        ),
        (
            vec![horizon - 1e-13, horizon - 1e-14, horizon],
            "1e-13 .. exactly T",
        ),
        (vec![horizon - 1e-9, horizon - 1e-10], "1e-9 .. 1e-10"),
    ] {
        for (baseline, excitation, decay) in [(0.5, 0.6, 1.0), (2.0, 0.9, 5.0), (0.05, 0.05, 0.1)] {
            let parameters = Parameters::new(baseline, excitation, decay).unwrap();
            let observation = Observation::new(&times, horizon).unwrap();
            assert_bit_identical(
                &parameters,
                &observation,
                &format!("{label} at ({baseline}, {excitation}, {decay})"),
            );
        }
    }
}

/// Long realizations, where any per-event divergence has the most room to accumulate
/// into something a tolerance would still accept but the bit pattern will not.
#[test]
fn agree_on_long_realizations() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    for (count, decay) in [(2_000usize, 0.05), (2_000, 4.0), (20_000, 1.0)] {
        let horizon = count as f64 / 2.0;
        let mut times: Vec<f64> = (0..count).map(|_| next() * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let parameters = Parameters::new(1.3, 0.6, decay).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();
        assert_bit_identical(
            &parameters,
            &observation,
            &format!("n={count} beta={decay}"),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn agree_over_random_parameters_and_events(
        baseline in 0.01f64..8.0,
        excitation in 0.001f64..0.99,
        decay in 0.01f64..8.0,
        horizon in 1.0f64..200.0,
        raw in prop::collection::vec(0.0f64..1.0, 0..200),
    ) {
        let mut times: Vec<f64> = raw.iter().map(|u| u * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let parameters = Parameters::new(baseline, excitation, decay).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();

        let value_only = negative_log_likelihood(&parameters, &observation);
        let (with_gradient, _) = negative_log_likelihood_and_gradient(&parameters, &observation);
        prop_assert_eq!(
            value_only.to_bits(),
            with_gradient.to_bits(),
            "value-only {:?} vs value+gradient {:?}, difference {:e}",
            value_only,
            with_gradient,
            (value_only - with_gradient).abs()
        );
    }

    /// Repeated timestamps drawn from a small grid, so ties are common rather than
    /// incidental.
    #[test]
    fn agree_when_ties_are_common(
        baseline in 0.05f64..3.0,
        excitation in 0.01f64..0.95,
        decay in 0.05f64..5.0,
        grid in prop::collection::vec(0u32..12, 1..60),
    ) {
        let horizon = 12.0;
        let mut times: Vec<f64> = grid.iter().map(|g| f64::from(*g)).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let parameters = Parameters::new(baseline, excitation, decay).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();

        let value_only = negative_log_likelihood(&parameters, &observation);
        let (with_gradient, _) = negative_log_likelihood_and_gradient(&parameters, &observation);
        prop_assert_eq!(value_only.to_bits(), with_gradient.to_bits());
    }
}
