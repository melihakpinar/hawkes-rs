//! The `O(n)` recursion against the `O(n^2)` definition (M1 Part B step 7).
//!
//! This is the primary correctness gate for the likelihood, and it uses no `tick`.
//! The reference is validated separately in `reference_loglikelihood.rs` against hand
//! calculations and the Poisson degenerate case, so this comparison is not circular.
//!
//! # Sabotage
//!
//! Replacing the grouped recursion (4.4) with the textbook form [Laub2015, eq. 20]
//! turned `agrees_with_brute_force_on_tied_input` red while leaving the
//! distinct-timestamp cases green — the intended signature of that bug. Dropping the
//! `count_at_previous_time` term, and advancing the state with `1.0` instead, did the
//! same. Recorded in `docs/verification-log.md`.

mod common;

use common::{
    RECURSION_TOLERANCE, brute_force_negative_log_likelihood as brute_force, computation_scale,
};
use hawk::univariate::{Observation, Parameters, negative_log_likelihood};
use proptest::prelude::*;

/// Upper bound on `n` for the comparison.
///
/// The recursion carries its state across the whole sequence, so its error grows —
/// slowly, damped by `exp(-beta*d_j)` at every step, and no faster than `sqrt(n)`.
/// At the measured `1.3e-14` for `n = 20000`, `n = 50000` predicts about `2e-14`,
/// still roughly 50x inside the gate; reaching `1e-12` would need `n` of order
/// `10^8`. The bound exists so the claim is checked rather than assumed.
///
/// The `O(n^2)` reference is *not* the limiting side, which is counterintuitive
/// enough to be worth stating: its error against exactly-rounded summation measured
/// zero at every `n` tested. The inner sum's terms decay geometrically, so only
/// `O(1/(beta*inter-arrival))` of them are numerically significant and its effective
/// length does not grow with `n`. That argument is specific to a geometrically
/// decaying kernel and must be redone if a heavy-tailed one is ever added.
const MAX_EVENTS_FOR_COMPARISON: usize = 50_000;

/// Compares against the **scale of the computation**, not against `|nll|`.
///
/// `nll = mu*T + alpha*compensator - log_term` is a difference of large terms and
/// passes through zero. Dividing by `|nll|` makes the relative error diverge wherever
/// those terms cancel — on correct code — and a randomized sweep reaches that region,
/// because it is a surface through the parameter space rather than a corner of it.
/// See `computation_scale` and `univariate_loglikelihood.md` §5.
fn assert_agrees(parameters: &Parameters, observation: &Observation, context: &str) {
    let recursive = negative_log_likelihood(parameters, observation);
    let reference = brute_force(parameters, observation);
    let scale = computation_scale(parameters, observation);
    let discrepancy = (recursive - reference).abs();
    assert!(
        discrepancy <= RECURSION_TOLERANCE * scale,
        "{context}: recursion {recursive:?} vs definition {reference:?}, \
         |difference| {discrepancy:e} > {RECURSION_TOLERANCE:e} * scale {scale:e}"
    );
}

#[test]
fn agrees_with_brute_force_on_tied_input() {
    // The case the textbook recursion gets wrong. Grouped by distinct time, a tie
    // must not excite itself; the definition computes that directly.
    for (times, horizon) in [
        (vec![1.0, 2.0, 2.0, 3.0], 5.0),
        (vec![1.0, 2.0, 2.0, 2.0, 3.0], 5.0),
        (vec![0.0, 0.0, 1.5, 3.0, 5.0, 5.0], 5.0),
        (vec![2.0, 2.0, 2.0, 2.0], 4.0),
        (vec![0.0, 0.0], 1.0),
    ] {
        let parameters = Parameters::new(0.7, 0.5, 1.3).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();
        assert_agrees(&parameters, &observation, &format!("{times:?}"));
    }
}

#[test]
fn agrees_with_brute_force_on_degenerate_input() {
    let parameters = Parameters::new(0.9, 0.4, 1.7).unwrap();
    for (times, horizon) in [
        (vec![], 5.0),
        (vec![2.5], 5.0),
        (vec![0.0], 5.0),
        (vec![5.0], 5.0),
        (vec![0.0, 5.0], 5.0),
    ] {
        let observation = Observation::new(&times, horizon).unwrap();
        assert_agrees(&parameters, &observation, &format!("{times:?}"));
    }
}

/// Long realizations, where the recursion has the most state to carry.
#[test]
fn agrees_with_brute_force_on_long_realizations() {
    // Deterministic pseudo-random times: a small LCG keeps this test independent of
    // the simulator, which is validated separately and must not be a prerequisite
    // for the likelihood gate.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    for (count, decay) in [(500usize, 0.9), (2000, 0.05), (2000, 4.0)] {
        assert!(count <= MAX_EVENTS_FOR_COMPARISON);
        let horizon = count as f64 / 2.0;
        let mut times: Vec<f64> = (0..count).map(|_| next() * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let parameters = Parameters::new(1.3, 0.6, decay).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();
        assert_agrees(
            &parameters,
            &observation,
            &format!("n={count} beta={decay}"),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Randomized sweep. This is the test that walks into the cancellation region
    /// where `|nll|` is near zero, which is why the gate divides by the computation
    /// scale instead.
    #[test]
    fn agrees_with_brute_force_over_random_parameters(
        baseline in 0.05f64..5.0,
        excitation in 0.01f64..0.95,
        decay in 0.05f64..5.0,
        horizon in 1.0f64..50.0,
        raw in prop::collection::vec(0.0f64..1.0, 0..60),
    ) {
        let mut times: Vec<f64> = raw.iter().map(|u| u * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let parameters = Parameters::new(baseline, excitation, decay).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();

        let recursive = negative_log_likelihood(&parameters, &observation);
        let reference = brute_force(&parameters, &observation);
        let scale = computation_scale(&parameters, &observation);
        let discrepancy = (recursive - reference).abs();
        prop_assert!(
            discrepancy <= RECURSION_TOLERANCE * scale,
            "recursion {recursive:?} vs definition {reference:?}, |difference| \
             {discrepancy:e} > {RECURSION_TOLERANCE:e} * scale {scale:e} \
             (nll is {recursive:e}, so a gate relative to |nll| would read \
             {:e})",
            discrepancy / recursive.abs().max(f64::MIN_POSITIVE),
        );
    }
}
