//! Round-trip property test (CLAUDE.md §3, oracle 3; M1 Part B steps 10 and 11).
//!
//! Random valid parameters -> simulate -> fit -> the parameters must come back. This
//! is the main regression net, and the only test that exercises the simulator, the
//! likelihood, the gradient and the optimizer as one system.
//!
//! The M0 stubs are gone: `stub_simulate_and_fit` has been replaced by
//! [`hawk::univariate::simulate`] and [`hawk::univariate::fit`].
//!
//! # Ties are deliberately not generated
//!
//! Ogata thinning draws continuous inter-arrival times, so it produces ties with
//! probability zero, and this test never synthesises timestamps directly. That is
//! not an accident of the strategy — it is required. On tied data the objective is
//! not a likelihood at all ([Laub2015, Theorem 3] is stated for a *simple* point
//! process, where simultaneous arrivals have probability zero), so the
//! maximum-likelihood asymptotics this test's tolerance is derived from do not hold.
//! Feeding tied data in would produce failures that look like estimator defects and
//! are not. See `docs/derivations/univariate_loglikelihood.md` §3.1.
//!
//! Tie handling is verified elsewhere and thoroughly: against the brute-force
//! definition in `loglikelihood.rs`, and against `tick` via four hand-built tied
//! fixtures in `differential_tick.rs`.
//!
//! # Sabotage
//!
//! Returning `truth` unchanged from the simulate-and-fit round trip — the M0 stub's
//! behaviour — makes `recovers_parameters_within_their_standard_errors` pass
//! vacuously, which is why the test also asserts the fit actually moved from its
//! starting point. Perturbing the fitted baseline by 5% turned it red. Recorded in
//! `docs/verification-log.md`.

mod common;

use common::asymptotic_standard_errors;
use hawk::univariate::{Observation, Parameters, fit, simulate};
use proptest::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// How many standard errors the estimate may sit from the truth.
///
/// **Derived, not chosen to make the test pass.** For each realization the test
/// computes the asymptotic standard error of the maximum-likelihood estimate from the
/// observed Fisher information — the Hessian of the negative log-likelihood at the
/// optimum, `Var(theta_hat) ~= I(theta_hat)^-1` — and requires the estimate to lie
/// within `TOLERANCE_IN_STANDARD_ERRORS` of the truth. The scale therefore adapts to
/// each realization: a short, low-intensity sample is allowed a wide interval and a
/// long one is held to a narrow one, which a fixed relative tolerance cannot do.
///
/// 5 rather than 2 or 3 for two reasons. The asymptotic normality is approximate at
/// these sample sizes, and the tails are the part of the approximation that is worst.
/// And the test makes 3 assertions on each of 200 cases: at 3 standard errors and
/// exact normality the expected number of spurious failures would be
/// `600 * 0.0027 = 1.6`, i.e. a test that fails most runs. At 5 it is `600 * 5.7e-7`,
/// about 1 run in 3000.
///
/// This is not a loose tolerance in the usual sense. A systematically wrong estimator
/// misses by many standard errors — the sabotage below moves the baseline by 5%,
/// which on these samples is tens of standard errors — while a correct one misses by
/// order 1.
const TOLERANCE_IN_STANDARD_ERRORS: f64 = 5.0;

/// Realizations shorter than this cannot support the asymptotics the tolerance rests
/// on, so they are discarded rather than tested. Simulation is random, so an
/// unusually quiet draw is possible even with generous parameters.
const MINIMUM_EVENTS: usize = 200;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// The main regression net.
    #[test]
    fn recovers_parameters_within_their_standard_errors(
        baseline in 0.3f64..2.0,
        // Bounded away from 0: as alpha -> 0 the kernel vanishes and beta stops being
        // identifiable at all, so the information matrix becomes singular and the
        // test would be vacuous rather than passing.
        excitation in 0.2f64..0.8,
        decay in 0.4f64..3.0,
        seed in 0u64..100_000,
    ) {
        let truth = Parameters::new(baseline, excitation, decay).unwrap();
        // Long enough that the asymptotics are reasonable. The expected count is
        // horizon * mu/(1-alpha), which is at least 3000 * 0.3/0.8 ~ 1100.
        let horizon = 3000.0;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let times = simulate(&truth, horizon, &mut rng).unwrap();
        prop_assume!(times.len() >= MINIMUM_EVENTS);

        let observation = Observation::new(&times, horizon).unwrap();
        let fitted = fit(&observation).unwrap();

        // The fit must be at least as good as the truth, or the optimizer has not
        // done its job -- this holds regardless of sampling noise and needs no
        // tolerance.
        let at_truth = hawk::univariate::negative_log_likelihood(&truth, &observation);
        prop_assert!(
            fitted.negative_log_likelihood <= at_truth + 1e-6,
            "fit found {:?} but the true parameters give {at_truth:?}, which is better",
            fitted.negative_log_likelihood
        );

        let errors = asymptotic_standard_errors(&fitted.parameters, &observation);
        prop_assume!(errors.is_some());
        let errors = errors.unwrap();

        for (index, (name, estimate, truth_value)) in [
            ("baseline", fitted.parameters.baseline(), baseline),
            ("excitation", fitted.parameters.excitation(), excitation),
            ("decay", fitted.parameters.decay(), decay),
        ].into_iter().enumerate() {
            let deviation = (estimate - truth_value).abs() / errors[index];
            prop_assert!(
                deviation <= TOLERANCE_IN_STANDARD_ERRORS,
                "{name}: fitted {estimate:?}, truth {truth_value:?}, standard error \
                 {:e} -- off by {deviation:?} standard errors, limit \
                 {TOLERANCE_IN_STANDARD_ERRORS} (n = {}, seed = {seed})",
                errors[index],
                times.len(),
            );
        }
    }
}

/// Guards against the round trip passing vacuously.
///
/// The M0 stub returned its input unchanged, which satisfies any recovery predicate.
/// The real fitter starts from a moment-matched guess and must move away from it, so
/// this asserts the optimizer did work rather than echoing either its input or its
/// starting point.
#[test]
fn the_fit_actually_optimizes() {
    let truth = Parameters::new(0.8, 0.5, 1.5).unwrap();
    let horizon = 3000.0;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let times = simulate(&truth, horizon, &mut rng).unwrap();
    let observation = Observation::new(&times, horizon).unwrap();

    // The starting point, reproduced from `fit`'s documented construction.
    let rate = times.len() as f64 / horizon;
    let start = Parameters::new(0.5 * rate, 0.5, times.len() as f64 / horizon).unwrap();
    let at_start = hawk::univariate::negative_log_likelihood(&start, &observation);

    let fitted = fit(&observation).unwrap();
    assert!(
        fitted.negative_log_likelihood < at_start,
        "the fit did not improve on its starting point: {:?} vs {at_start:?}",
        fitted.negative_log_likelihood
    );
    assert!(fitted.iterations > 0, "the optimizer took no iterations");
    assert!(fitted.converged, "the optimizer hit the iteration cap");
    assert!(
        fitted.is_stationary(),
        "a fit to stationary data reported branching ratio {}",
        fitted.branching_ratio()
    );
}

#[test]
fn rejects_data_that_cannot_identify_the_parameters() {
    let observation = Observation::new(&[1.0, 2.0], 5.0).unwrap();
    assert!(
        fit(&observation).is_err(),
        "fitting three parameters to two events must be an error"
    );
    let empty = Observation::new(&[], 5.0).unwrap();
    assert!(
        fit(&empty).is_err(),
        "fitting to no events must be an error"
    );
}
