//! Multivariate round-trip property test (M2 Part B step 13).
//!
//! Random valid parameters -> simulate -> fit -> the parameters must come back,
//! **elementwise** on the excitation matrix.
//!
//! # Why elementwise
//!
//! A transposed excitation matrix has the same entries, the same Frobenius norm, the
//! same spectral radius and the same row-sum multiset when the matrix is close to
//! symmetric. Any aggregate comparison — total error, norm of the difference, spectral
//! radius agreement — can pass on a transposition. Comparing `alpha[i][j]` against
//! `truth[i][j]` for every pair is the only form that cannot, and it is the reason
//! `conventions.md` C6 is worth having pinned.
//!
//! `transposition_is_caught_by_the_elementwise_comparison` proves that on the actual
//! recovered matrices rather than asserting it.
//!
//! # Ties are not generated
//!
//! Ogata thinning gives ties probability zero and this test never synthesises
//! timestamps. That is required, not incidental: on tied data the objective is not a
//! likelihood (`multivariate_loglikelihood.md` §3.1), so the asymptotics the tolerance
//! is derived from do not hold.
//!
//! # Sabotage
//!
//! Treating the log coordinate for `excitation` as if it were natural — dropping the
//! `exp` in `fit`'s parameter reconstruction — turned the recovery red on every case.
//! Transposing the fitted matrix before comparison turned
//! `recovers_the_excitation_matrix_elementwise` red and left a Frobenius-norm
//! comparison green, which is the point of the test. Recorded in
//! `docs/verification-log.md`.

// Index loops over components and matrix entries; the indices are the content, for
// the reason given at the top of `hawkes/src/multivariate.rs`.
#![allow(clippy::needless_range_loop)]

mod common;

use common::multivariate_asymptotic_standard_errors as standard_errors;
use hawkes::multivariate::{
    Observation, Parameters, fit, fit_from, negative_log_likelihood, simulate,
};
use proptest::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// How many standard errors an estimate may sit from the truth.
///
/// Derived, as in M1: for each realization the asymptotic standard error comes from
/// the observed Fisher information, and the estimate must lie within this many of
/// them. `6` rather than M1's `5` because a `d`-component fit makes `d + d^2 + 1`
/// assertions per case — up to 31 at `d = 5` — so with 120 cases there are thousands
/// of them, and the tail of the normal approximation has to absorb that. At 6 the
/// expected count of spurious failures across the whole test is below 0.01.
///
/// A systematically wrong estimator misses by far more: the transposition below moves
/// entries by tens of standard errors.
const TOLERANCE_IN_STANDARD_ERRORS: f64 = 6.0;

const MINIMUM_EVENTS_PER_COMPONENT: usize = 150;

fn truth_for(d: usize, seed: u64) -> Parameters {
    let mut rng = common::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let baseline: Vec<f64> = (0..d).map(|_| 0.25 + rng.next_f64() * 0.8).collect();
    // Row sums bounded well below 1, so the process is comfortably stationary and the
    // realization is not dominated by one component.
    let excitation: Vec<f64> = (0..d * d)
        .map(|_| 0.05 + rng.next_f64() * 0.55 / d as f64)
        .collect();
    let decay = 0.8 + rng.next_f64() * 1.4;
    Parameters::new(baseline, excitation, decay).unwrap()
}

#[test]
fn transposition_is_caught_by_the_elementwise_comparison() {
    // A concrete demonstration that the aggregate comparisons this test avoids would
    // not have caught a transposition.
    let d = 3;
    let truth = truth_for(d, 7);
    let mut transposed = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            transposed[i * d + j] = truth.excitation_at(j, i);
        }
    }

    let frobenius_truth: f64 = truth.excitation().iter().map(|v| v * v).sum::<f64>().sqrt();
    let frobenius_transposed: f64 = transposed.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (frobenius_truth - frobenius_transposed).abs() < 1e-15,
        "a transpose preserves the Frobenius norm exactly, which is why a norm-based \
         recovery test cannot see it"
    );

    let transposed_parameters =
        Parameters::new(truth.baseline().to_vec(), transposed, truth.decay()).unwrap();
    assert!(
        (truth.branching_ratio_spectral_radius()
            - transposed_parameters.branching_ratio_spectral_radius())
        .abs()
            < 1e-9,
        "a transpose preserves the spectral radius, so that diagnostic cannot see it \
         either"
    );

    // Elementwise, it is obvious.
    let differing = (0..d * d)
        .filter(|&k| (truth.excitation()[k] - transposed_parameters.excitation()[k]).abs() > 1e-9)
        .count();
    assert!(
        differing >= 4,
        "the test matrix is too close to symmetric for a transposition to be \
         detectable; only {differing} of {} entries differ",
        d * d
    );
}

#[test]
fn multi_start_invariance() {
    // A fit that lands somewhere different depending on where it began is reporting a
    // local optimum, and the round trip would be measuring the starting point.
    let d = 3;
    let truth = truth_for(d, 11);
    let horizon = 4000.0;
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let events = simulate(&truth, horizon, &mut rng).unwrap();
    let observation = Observation::new(&events, horizon).unwrap();

    let reference = fit(&observation).expect("default start");
    assert!(reference.converged, "default start did not converge");

    let n = d + d * d + 1;
    for (label, scale) in [("low", 0.2f64), ("high", 3.0), ("very low", 0.05)] {
        let start: Vec<f64> = (0..n)
            .map(|k| {
                let base = if k < d {
                    0.5
                } else if k < d + d * d {
                    0.5 / d as f64
                } else {
                    1.0
                };
                (base * scale).ln()
            })
            .collect();
        let alternative = fit_from(&observation, start).expect("alternative start");
        assert!(
            alternative.converged,
            "{label} start did not converge (gradient norm {:e})",
            alternative.gradient_norm
        );
        // Same optimum to well inside sampling noise: the objective is what the
        // optimizer is minimising, so comparing it is the sharpest check.
        let gap = (alternative.negative_log_likelihood - reference.negative_log_likelihood).abs();
        assert!(
            gap <= 1e-6 * reference.negative_log_likelihood.abs().max(1.0),
            "{label} start reached a different optimum: {:?} vs {:?}",
            alternative.negative_log_likelihood,
            reference.negative_log_likelihood
        );
    }
}

#[test]
fn a_true_zero_entry_is_recovered_as_a_small_positive_number() {
    // The boundary case `docs/derivations/parameter_space.md` is about: log space
    // cannot return an exact zero, and the estimate should sit within its own standard
    // error of zero rather than at some arbitrary floor.
    let d = 2;
    let truth = Parameters::new(vec![0.5, 0.4], vec![0.25, 0.0, 0.30, 0.20], 1.2).unwrap();
    let horizon = 6000.0;
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let events = simulate(&truth, horizon, &mut rng).unwrap();
    let observation = Observation::new(&events, horizon).unwrap();
    let fitted = fit(&observation).unwrap();

    let estimate = fitted.parameters.excitation_at(0, 1);
    assert!(estimate > 0.0, "log space cannot return an exact zero");

    // Small relative to the entries that are genuinely non-zero. This holds without
    // any distributional assumption, and it is the property a caller thresholding a
    // fitted matrix actually depends on.
    let genuine = fitted.parameters.excitation_at(0, 0);
    assert!(
        estimate < 0.05 * genuine,
        "the true-zero entry was recovered as {estimate:?}, not small against the \
         genuinely non-zero entry {genuine:?}"
    );

    // And within its own standard error of zero, when the information matrix is
    // usable. It need not be: a coordinate pinned near the boundary can make the
    // Hessian ill-conditioned, which is a fact about the data rather than a defect,
    // so the assertion above is the one that always applies.
    if let Some(errors) = standard_errors(&fitted.parameters, &observation) {
        let error = errors[d + 1]; // flat layout: baseline (d), then excitation[0][1]
        assert!(
            estimate <= TOLERANCE_IN_STANDARD_ERRORS * error,
            "a true zero was recovered as {estimate:?}, which is {:?} standard \
             errors from zero (standard error {error:e})",
            estimate / error
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(120))]

    #[test]
    fn recovers_the_excitation_matrix_elementwise(
        d in 2usize..=5,
        seed in 0u64..50_000,
    ) {
        let truth = truth_for(d, seed);
        prop_assume!(truth.is_stationary());
        let horizon = 5000.0;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let events = simulate(&truth, horizon, &mut rng).unwrap();
        prop_assume!(events.iter().all(|c| c.len() >= MINIMUM_EVENTS_PER_COMPONENT));

        let observation = Observation::new(&events, horizon).unwrap();
        let fitted = fit(&observation).unwrap();

        // Tolerance-free: the fit must be at least as good as the truth, or the
        // optimizer has not done its job.
        let at_truth = negative_log_likelihood(&truth, &observation).unwrap();
        prop_assert!(
            fitted.negative_log_likelihood <= at_truth + 1e-6,
            "fit found {:?} but the truth gives {:?}",
            fitted.negative_log_likelihood, at_truth
        );

        let errors = standard_errors(&fitted.parameters, &observation);
        prop_assume!(errors.is_some());
        let errors = errors.unwrap();

        for i in 0..d {
            let deviation = (fitted.parameters.baseline()[i] - truth.baseline()[i]).abs()
                / errors[i];
            prop_assert!(deviation <= TOLERANCE_IN_STANDARD_ERRORS,
                "baseline[{}]: fitted {:?} truth {:?}, {:?} standard errors (d={}, seed={})",
                i, fitted.parameters.baseline()[i], truth.baseline()[i], deviation, d, seed);
        }
        // ELEMENTWISE. A transposed matrix fails here and passes any aggregate check.
        for i in 0..d {
            for j in 0..d {
                let estimate = fitted.parameters.excitation_at(i, j);
                let actual = truth.excitation_at(i, j);
                let deviation = (estimate - actual).abs() / errors[d + i * d + j];
                prop_assert!(deviation <= TOLERANCE_IN_STANDARD_ERRORS,
                    "excitation[{}][{}]: fitted {:?} truth {:?}, {:?} standard errors \
                     (d={}, seed={}). Transposed truth here is {:?}.",
                    i, j, estimate, actual, deviation, d, seed,
                    truth.excitation_at(j, i));
            }
        }
        let deviation = (fitted.parameters.decay() - truth.decay()).abs() / errors[d + d * d];
        prop_assert!(deviation <= TOLERANCE_IN_STANDARD_ERRORS,
            "decay: fitted {:?} truth {:?}, {:?} standard errors",
            fitted.parameters.decay(), truth.decay(), deviation);

        // Spectral radius as a diagnostic, not a constraint.
        prop_assert!(fitted.branching_ratio_spectral_radius().is_finite());
    }
}
