//! The analytic gradient against central differences (M1 Part B step 8).
//!
//! Uses the same checker the M0 harness is built on — `central_difference_gradient`
//! and `max_relative_discrepancy` in `common/` — which `gradient_check.rs` has
//! already proven can go red, on closed-form functions with known gradients.
//!
//! `d/dbeta` (G.7) is the partial that matters most here. `tick` cannot check it at
//! all: `ModelHawkesExpKernLogLik` takes `decay` as a fixed constructor argument
//! rather than a coefficient, so it exposes no derivative with respect to it. (G.1)
//! and (G.2) were independently confirmed against `tick` in M0's experiment E2;
//! (G.7) rests on the derivation plus this test alone.
//!
//! # Sabotage
//!
//! Dropping `beta * excitation_state_derivative` from (G.4) — the term the derivation
//! singles out as the one most likely to be omitted — turned
//! `analytic_gradient_matches_central_differences` red on `decay` while leaving
//! `baseline` and `excitation` green. Computing `Bp` from the pre-update state
//! instead of the advanced one (hazard 1 of the gradient derivation §5) did the same.
//! Removing the `parameters.decay *` factor from `to_log_parameter_space` turned only
//! the log-space test red. Recorded in `docs/verification-log.md`.

mod common;

use common::{STEP, central_difference_gradient, max_relative_discrepancy};
use hawkes::univariate::{
    Observation, Parameters, negative_log_likelihood, negative_log_likelihood_and_gradient,
};
use proptest::prelude::*;

/// Agreement required between the analytic gradient and central differences.
///
/// Looser than `gradient_check.rs`'s `1e-7`, which is measured on closed-form
/// functions of two or three variables. Here the function is a sum of `n` logarithms
/// and `n` exponentials, so the central difference's own round-off floor is larger:
/// its numerator differences two evaluations that each carry `O(sqrt(n) * eps)`
/// relative error. `1e-6` sits above that floor and well below anything a real
/// derivative bug produces — the sabotage cases below move `d/dbeta` by whole
/// percent, not by parts per million.
const TOLERANCE: f64 = 1e-6;

fn nll_at(natural: &[f64], times: &[f64], horizon: f64) -> f64 {
    let parameters = Parameters::new(natural[0], natural[1], natural[2])
        .expect("generated parameters must be valid");
    let observation = Observation::new(times, horizon).expect("times must be valid");
    negative_log_likelihood(&parameters, &observation)
}

fn check(times: &[f64], horizon: f64, baseline: f64, excitation: f64, decay: f64, label: &str) {
    let parameters = Parameters::new(baseline, excitation, decay).unwrap();
    let observation = Observation::new(times, horizon).unwrap();
    let (_, analytic) = negative_log_likelihood_and_gradient(&parameters, &observation);

    // Natural parameters.
    let point = [baseline, excitation, decay];
    let numeric = central_difference_gradient(|v| nll_at(v, times, horizon), &point, STEP);
    let analytic_vec = vec![analytic.baseline, analytic.excitation, analytic.decay];
    let discrepancy = max_relative_discrepancy(&analytic_vec, &numeric);
    assert!(
        discrepancy <= TOLERANCE,
        "{label}: natural-space gradient {analytic_vec:?} vs central differences \
         {numeric:?}, max relative discrepancy {discrepancy:e} > {TOLERANCE:e}"
    );

    // Log-parameter space (G.8). Checked separately because the chain-rule factor is
    // exactly where one can go missing, and the natural-space check would not see it.
    let log_point = [baseline.ln(), excitation.ln(), decay.ln()];
    let log_numeric = central_difference_gradient(
        |v| nll_at(&[v[0].exp(), v[1].exp(), v[2].exp()], times, horizon),
        &log_point,
        STEP,
    );
    let log_analytic = analytic.to_log_parameter_space(&parameters);
    let log_vec = vec![
        log_analytic.baseline,
        log_analytic.excitation,
        log_analytic.decay,
    ];
    let log_discrepancy = max_relative_discrepancy(&log_vec, &log_numeric);
    assert!(
        log_discrepancy <= TOLERANCE,
        "{label}: log-space gradient {log_vec:?} vs central differences \
         {log_numeric:?}, max relative discrepancy {log_discrepancy:e} > {TOLERANCE:e}"
    );
}

#[test]
fn matches_on_tied_input() {
    // The excitation-state derivative Bp_j is advanced only at a change of distinct
    // time, so ties exercise a path the distinct-timestamp cases never reach.
    for (times, horizon) in [
        (vec![1.0, 2.0, 2.0, 3.0], 5.0),
        (vec![1.0, 2.0, 2.0, 2.0, 3.5], 6.0),
        (vec![0.0, 0.0, 1.5, 3.0, 5.0, 5.0], 5.0),
    ] {
        check(&times, horizon, 0.7, 0.5, 1.3, &format!("{times:?}"));
    }
}

#[test]
fn matches_on_degenerate_input() {
    // n == 0 returns (T, 0, 0) by a separate branch, so it is checked explicitly
    // rather than left to the sweep.
    let parameters = Parameters::new(0.8, 0.4, 1.2).unwrap();
    let observation = Observation::new(&[], 5.0).unwrap();
    let (nll, gradient) = negative_log_likelihood_and_gradient(&parameters, &observation);
    assert!((nll - 0.8 * 5.0).abs() < 1e-15, "empty: nll should be mu*T");
    assert!(
        (gradient.baseline - 5.0).abs() < 1e-15,
        "empty: d/dmu should be T"
    );
    assert_eq!(gradient.excitation, 0.0, "empty: d/dalpha should be 0");
    assert_eq!(gradient.decay, 0.0, "empty: d/dbeta should be 0");

    for (times, horizon) in [(vec![2.5], 5.0), (vec![0.0], 5.0), (vec![5.0], 5.0)] {
        check(&times, horizon, 0.8, 0.4, 1.2, &format!("{times:?}"));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn analytic_gradient_matches_central_differences(
        baseline in 0.2f64..3.0,
        excitation in 0.05f64..0.9,
        decay in 0.2f64..3.0,
        horizon in 2.0f64..30.0,
        raw in prop::collection::vec(0.0f64..1.0, 0..40),
    ) {
        let mut times: Vec<f64> = raw.iter().map(|u| u * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let parameters = Parameters::new(baseline, excitation, decay).unwrap();
        let observation = Observation::new(&times, horizon).unwrap();
        let (_, analytic) = negative_log_likelihood_and_gradient(&parameters, &observation);

        let point = [baseline, excitation, decay];
        let numeric = central_difference_gradient(
            |v| nll_at(v, &times, horizon), &point, STEP);
        let analytic_vec = vec![analytic.baseline, analytic.excitation, analytic.decay];
        let discrepancy = max_relative_discrepancy(&analytic_vec, &numeric);
        prop_assert!(
            discrepancy <= TOLERANCE,
            "gradient {analytic_vec:?} vs central differences {numeric:?}, \
             max relative discrepancy {discrepancy:e} > {TOLERANCE:e}"
        );

        let log_point = [baseline.ln(), excitation.ln(), decay.ln()];
        let log_numeric = central_difference_gradient(
            |v| nll_at(&[v[0].exp(), v[1].exp(), v[2].exp()], &times, horizon),
            &log_point, STEP);
        let log_analytic = analytic.to_log_parameter_space(&parameters);
        let log_vec = vec![
            log_analytic.baseline, log_analytic.excitation, log_analytic.decay];
        let log_discrepancy = max_relative_discrepancy(&log_vec, &log_numeric);
        prop_assert!(
            log_discrepancy <= TOLERANCE,
            "log-space gradient {log_vec:?} vs central differences {log_numeric:?}, \
             max relative discrepancy {log_discrepancy:e} > {TOLERANCE:e}"
        );
    }
}
