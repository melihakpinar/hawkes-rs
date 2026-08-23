//! The multivariate analytic gradient against central differences (M2 Part B step 11).
//!
//! Uses the same checker the M0 harness is built on, which `gradient_check.rs` has
//! already proven can go red.
//!
//! Two things here that the univariate check could not express:
//!
//! - **Index orientation.** `d nll / d alpha[i][j]` involves `E_j`, the events of `i`,
//!   and the state of `j` (`multivariate_gradient.md` §3). A transposition survives on
//!   symmetric data, so the cases below are deliberately asymmetric.
//! - **Partial omission of the `beta*Bp` term.** In M1 there was one such term; here
//!   there are `d^2`, and dropping a subset leaves a gradient wrong only in the
//!   directions involving those pairs.
//!
//! `tick` cannot check any of `d/dbeta`: `decay` is a fixed constructor argument
//! there, at every `d`.
//!
//! # Sabotage
//!
//! Transposing the excitation index in the gradient turned the asymmetric cases red
//! and left the symmetric ones green. Dropping `beta * state_derivative` from the
//! pair accumulator turned `decay` red. Computing the state derivative from the
//! pre-update state did the same. Removing the chain-rule factor for `excitation`
//! turned only the log-space assertions red. Recorded in
//! `docs/verification-log.md`.

mod common;

use common::{Lcg, STEP, central_difference_gradient, max_relative_discrepancy};
use hawk::multivariate::{
    Observation, Parameters, negative_log_likelihood, negative_log_likelihood_and_gradient,
};
use proptest::prelude::*;

/// Same bound and justification as the univariate check: above the central
/// difference's own floor for a sum of `n` logarithms, far below anything a real
/// derivative bug produces.
const TOLERANCE: f64 = 1e-6;

/// Flat parameter layout for the finite-difference sweep:
/// `[baseline (d), excitation (d*d), decay (1)]`.
fn flatten(p: &Parameters) -> Vec<f64> {
    let mut flat = p.baseline().to_vec();
    flat.extend_from_slice(p.excitation());
    flat.push(p.decay());
    flat
}

fn unflatten(flat: &[f64], d: usize) -> Parameters {
    Parameters::new(
        flat[..d].to_vec(),
        flat[d..d + d * d].to_vec(),
        flat[d + d * d],
    )
    .expect("perturbed parameters must stay valid")
}

fn flat_gradient(p: &Parameters, observation: &Observation) -> Vec<f64> {
    let (_, gradient) = negative_log_likelihood_and_gradient(p, observation);
    let mut flat = gradient.baseline.clone();
    flat.extend_from_slice(&gradient.excitation);
    flat.push(gradient.decay);
    flat
}

fn flat_log_gradient(p: &Parameters, observation: &Observation) -> Vec<f64> {
    let (_, gradient) = negative_log_likelihood_and_gradient(p, observation);
    let log_gradient = gradient.to_log_parameter_space(p);
    let mut flat = log_gradient.baseline.clone();
    flat.extend_from_slice(&log_gradient.excitation);
    flat.push(log_gradient.decay);
    flat
}

/// Numeric gradient that is central in the interior and one-sided at the boundary.
///
/// `alpha[i][j] = 0` is a legitimate parameter value and a *boundary* of the domain:
/// a central difference would evaluate at `-h`, which `Parameters::new` rejects. The
/// derivative from the right still exists, so those coordinates use the second-order
/// one-sided formula
///
/// ```text
/// ( -3 f(x) + 4 f(x + h) - f(x + 2h) ) / (2h)
/// ```
///
/// rather than the first-order `(f(x+h) - f(x))/h`, whose `O(h)` error would be
/// `1e-5` at this step size and would not fit inside a `1e-6` gate.
fn boundary_aware_gradient<F>(f: F, point: &[f64], step: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut gradient = Vec::with_capacity(point.len());
    let mut probe = point.to_vec();
    let at_point = f(point);
    for index in 0..point.len() {
        let original = point[index];
        if original > 0.0 {
            probe[index] = original + step;
            let forward = f(&probe);
            probe[index] = original - step;
            let backward = f(&probe);
            gradient.push((forward - backward) / (2.0 * step));
        } else {
            probe[index] = original + step;
            let one = f(&probe);
            probe[index] = original + 2.0 * step;
            let two = f(&probe);
            gradient.push((-3.0 * at_point + 4.0 * one - two) / (2.0 * step));
        }
        probe[index] = original;
    }
    gradient
}

fn check(p: &Parameters, events: &[Vec<f64>], horizon: f64, label: &str) {
    let d = p.dimension();
    let observation = Observation::new(events, horizon).unwrap();

    let point = flatten(p);
    let numeric = boundary_aware_gradient(
        |v| {
            let perturbed = unflatten(v, d);
            let obs = Observation::new(events, horizon).unwrap();
            negative_log_likelihood(&perturbed, &obs)
        },
        &point,
        STEP,
    );
    let analytic = flat_gradient(p, &observation);
    let discrepancy = max_relative_discrepancy(&analytic, &numeric);
    assert!(
        discrepancy <= TOLERANCE,
        "{label}: natural-space gradient {analytic:?} vs central differences \
         {numeric:?}, max relative discrepancy {discrepancy:e} > {TOLERANCE:e}"
    );

    // Log-parameter space (MG.9), checked separately because the chain-rule factor is
    // invisible to the natural-space check. Excitation entries must be strictly
    // positive here for `ln` to exist; zero entries are a *fit* concern, addressed in
    // `docs/derivations/parameter_space.md`.
    if p.excitation().iter().all(|&a| a > 0.0) {
        let log_point: Vec<f64> = point.iter().map(|v| v.ln()).collect();
        let log_numeric = central_difference_gradient(
            |v| {
                let natural: Vec<f64> = v.iter().map(|x| x.exp()).collect();
                let perturbed = unflatten(&natural, d);
                let obs = Observation::new(events, horizon).unwrap();
                negative_log_likelihood(&perturbed, &obs)
            },
            &log_point,
            STEP,
        );
        let log_analytic = flat_log_gradient(p, &observation);
        let log_discrepancy = max_relative_discrepancy(&log_analytic, &log_numeric);
        assert!(
            log_discrepancy <= TOLERANCE,
            "{label}: log-space gradient {log_analytic:?} vs central differences \
             {log_numeric:?}, max relative discrepancy {log_discrepancy:e} > \
             {TOLERANCE:e}"
        );
    }
}

/// Strongly asymmetric, so a transposed index in (MG.2) cannot survive.
///
/// Exact zeros are used deliberately: they are ordinary in `d` dimensions, they sit on
/// the boundary of the domain, and `boundary_aware_gradient` is what makes them
/// checkable.
#[test]
fn matches_on_asymmetric_excitation() {
    let p = Parameters::new(
        vec![0.2, 0.5, 0.35],
        vec![0.05, 0.60, 0.00, 0.00, 0.05, 0.55, 0.40, 0.00, 0.05],
        1.2,
    )
    .unwrap();
    let events = vec![
        vec![1.0, 2.5, 4.0, 5.5],
        vec![0.5, 3.0, 4.5],
        vec![2.0, 3.5, 5.0],
    ];
    check(&p, &events, 6.0, "asymmetric d=3");
}

#[test]
fn matches_on_tied_input() {
    let p = Parameters::new(vec![0.7, 0.5], vec![0.3, 0.2, 0.1, 0.25], 1.3).unwrap();
    for events in [
        vec![vec![1.0, 2.0, 2.0, 3.0], vec![2.0, 4.0]],
        vec![vec![0.0, 0.0, 1.5], vec![0.0, 3.0, 3.0]],
        vec![vec![2.0; 4], vec![2.0, 5.0]],
    ] {
        check(&p, &events, 6.0, &format!("{events:?}"));
    }
}

#[test]
fn matches_on_degenerate_input() {
    // The all-empty branch returns its gradient without entering the loop.
    let p = Parameters::new(vec![0.8, 0.4], vec![0.2, 0.1, 0.05, 0.3], 1.2).unwrap();
    let empty = vec![vec![], vec![]];
    let observation = Observation::new(&empty, 5.0).unwrap();
    let (nll, gradient) = negative_log_likelihood_and_gradient(&p, &observation);
    assert!(
        (nll - (0.8 + 0.4) * 5.0).abs() < 1e-15,
        "empty: nll should be sum(mu)*T"
    );
    assert_eq!(
        gradient.baseline,
        vec![5.0, 5.0],
        "empty: d/dmu should be T"
    );
    assert_eq!(
        gradient.excitation,
        vec![0.0; 4],
        "empty: d/dalpha should be 0"
    );
    assert_eq!(gradient.decay, 0.0, "empty: d/dbeta should be 0");

    for events in [
        vec![vec![2.5], vec![]],
        vec![vec![], vec![2.5]],
        vec![vec![0.0], vec![5.0]],
    ] {
        check(&p, &events, 5.0, &format!("{events:?}"));
    }
}

/// Structural zeros in the excitation matrix are ordinary in `d` dimensions.
#[test]
fn matches_with_structural_zeros() {
    let events = vec![vec![0.5, 1.5, 3.0], vec![1.0, 2.0], vec![0.75, 2.5]];
    for (label, excitation) in [
        (
            "zero row",
            vec![0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2],
        ),
        (
            "zero column",
            vec![0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.1, 0.0],
        ),
    ] {
        let p = Parameters::new(vec![0.5, 0.6, 0.4], excitation, 1.1).unwrap();
        check(&p, &events, 5.0, label);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(150))]

    #[test]
    fn analytic_gradient_matches_central_differences(
        d in 1usize..=4,
        seed in 0u64..100_000,
        horizon in 3.0f64..25.0,
    ) {
        let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let events: Vec<Vec<f64>> = (0..d)
            .map(|_| {
                let n = rng.next_usize(20);
                let mut times: Vec<f64> =
                    (0..n).map(|_| rng.next_f64() * horizon).collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times
            })
            .collect();
        // Bounded away from zero so the log-space branch is exercised too.
        let baseline: Vec<f64> = (0..d).map(|_| 0.2 + rng.next_f64() * 2.0).collect();
        let excitation: Vec<f64> =
            (0..d * d).map(|_| 0.02 + rng.next_f64() * 0.7 / d as f64).collect();
        let decay = 0.3 + rng.next_f64() * 2.5;
        let p = Parameters::new(baseline, excitation, decay).unwrap();

        let observation = Observation::new(&events, horizon).unwrap();
        let point = flatten(&p);
        let numeric = central_difference_gradient(
            |v| {
                let perturbed = unflatten(v, d);
                let obs = Observation::new(&events, horizon).unwrap();
                negative_log_likelihood(&perturbed, &obs)
            },
            &point, STEP);
        let analytic = flat_gradient(&p, &observation);
        let discrepancy = max_relative_discrepancy(&analytic, &numeric);
        prop_assert!(discrepancy <= TOLERANCE,
            "d={} natural-space discrepancy {:e} > {:e}", d, discrepancy, TOLERANCE);

        let log_point: Vec<f64> = point.iter().map(|v| v.ln()).collect();
        let log_numeric = central_difference_gradient(
            |v| {
                let natural: Vec<f64> = v.iter().map(|x| x.exp()).collect();
                let perturbed = unflatten(&natural, d);
                let obs = Observation::new(&events, horizon).unwrap();
                negative_log_likelihood(&perturbed, &obs)
            },
            &log_point, STEP);
        let log_analytic = flat_log_gradient(&p, &observation);
        let log_discrepancy = max_relative_discrepancy(&log_analytic, &log_numeric);
        prop_assert!(log_discrepancy <= TOLERANCE,
            "d={} log-space discrepancy {:e} > {:e}", d, log_discrepancy, TOLERANCE);
    }
}
