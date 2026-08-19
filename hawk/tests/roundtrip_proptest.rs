//! Round-trip property harness (CLAUDE.md §3, oracle 3).
//!
//! The main regression net for the estimator: random valid parameters -> simulate
//! -> fit -> the fitted parameters must come back within tolerance.
//!
//! # M0 status
//!
//! `hawk` has neither a simulator nor an estimator, so [`stub_simulate_and_fit`]
//! short-circuits the loop and returns its input. What M0 tests is the harness:
//! that the generator only ever produces parameters a Hawkes process is actually
//! defined for, that `proptest` sweeps them, and that the recovery predicate fires.
//! The stub is deleted in M1.
//!
//! # Sabotage (CLAUDE.md §3)
//!
//! Replacing [`stub_simulate_and_fit`]'s body with a constant
//! `Parameters { baseline: 1.0, excitation: 0.5, decay: 1.0 }` turned
//! `simulate_then_fit_recovers_parameters` red immediately, and `proptest` shrank
//! the counterexample to the smallest generated parameter set. Recorded in
//! `docs/verification-log.md`.

use proptest::prelude::*;

/// Relative tolerance for parameter recovery.
///
/// M0 value only, and deliberately tight: the stub is exact, so anything looser
/// would not prove the predicate fires. M1 must replace this with a tolerance
/// justified by the estimator's actual sampling error at the horizons used, and
/// state that justification here.
const RECOVERY_RELATIVE_TOLERANCE: f64 = 1e-12;

/// Univariate exponential-kernel parameters.
///
/// Field names follow the conventions pinned in `docs/derivations/conventions.md`:
/// `excitation` is `alpha`, the *integral* of the kernel
/// `alpha * beta * exp(-beta t)` (C1), and therefore the branching ratio itself
/// (C2), not `alpha/beta`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Parameters {
    /// `mu` — the baseline intensity.
    baseline: f64,
    /// `alpha` — the branching ratio. Stationarity requires `alpha < 1` (C2).
    excitation: f64,
    /// `beta` — the exponential decay rate.
    decay: f64,
}

/// Generates parameters of a stationary univariate Hawkes process.
///
/// Bounds are chosen so that a simulation on a practical horizon terminates and
/// produces enough events to identify the parameters:
///
/// - `baseline` bounded away from 0, or long stretches produce no events at all;
/// - `excitation` capped at 0.9, since the expected cluster size is
///   `1/(1 - alpha)` and the process is non-stationary at 1.0;
/// - `decay` bounded away from 0, or the kernel has effectively infinite memory.
fn stationary_parameters() -> impl Strategy<Value = Parameters> {
    (0.05f64..5.0, 0.01f64..0.9, 0.1f64..5.0).prop_map(|(baseline, excitation, decay)| Parameters {
        baseline,
        excitation,
        decay,
    })
}

/// Stand-in for `simulate(params) -> events` then `fit(events) -> params`.
/// **Deleted in M1.**
fn stub_simulate_and_fit(truth: Parameters) -> Parameters {
    truth
}

fn relative_discrepancy(actual: f64, expected: f64) -> f64 {
    // Every parameter is strictly positive by construction, so a plain relative
    // error is well defined and no near-zero guard is needed.
    (actual - expected).abs() / expected.abs()
}

proptest! {
    #[test]
    fn simulate_then_fit_recovers_parameters(truth in stationary_parameters()) {
        let fitted = stub_simulate_and_fit(truth);

        for (name, actual, expected) in [
            ("baseline", fitted.baseline, truth.baseline),
            ("excitation", fitted.excitation, truth.excitation),
            ("decay", fitted.decay, truth.decay),
        ] {
            let discrepancy = relative_discrepancy(actual, expected);
            prop_assert!(
                discrepancy <= RECOVERY_RELATIVE_TOLERANCE,
                "{name}: fitted {actual:?}, truth {expected:?}, \
                 relative discrepancy {discrepancy:?} > {RECOVERY_RELATIVE_TOLERANCE:?}",
            );
        }
    }

    /// The generator is part of the harness, so it is tested too. A generator that
    /// emitted a non-stationary or non-positive parameter set would make every
    /// downstream failure ambiguous.
    #[test]
    fn generator_only_emits_stationary_parameters(p in stationary_parameters()) {
        prop_assert!(p.baseline > 0.0, "baseline {:?} is not positive", p.baseline);
        prop_assert!(p.decay > 0.0, "decay {:?} is not positive", p.decay);
        prop_assert!(p.excitation > 0.0, "excitation {:?} is not positive", p.excitation);
        prop_assert!(
            p.excitation < 1.0,
            "excitation {:?} is not stationary (conventions.md C2)",
            p.excitation
        );
    }
}
