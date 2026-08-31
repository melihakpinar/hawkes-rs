//! Finite-difference gradient harness (CLAUDE.md §3, oracle 4).
//!
//! A wrong derivative still converges — to the wrong place. Nothing else in the
//! verification suite catches that, so this harness has to be trustworthy before
//! there is a gradient to check.
//!
//! # M0 status
//!
//! Unlike the other two harnesses this one needs no stub. It is exercised against
//! closed-form test functions whose gradients are known by hand — a quadratic and a
//! separable transcendental — so it is fully tested on its own terms today, and M1
//! only has to point it at the log-likelihood.
//!
//! # Sabotage (CLAUDE.md §3)
//!
//! Confirmed to detect failure before being trusted. Negating a single component of
//! the analytic gradient in `central_difference_matches_quadratic` turned it red;
//! perturbing one component by `1e-6`, well below anything visible by inspection,
//! also turned it red. `detects_a_wrong_gradient` is the permanent form of that
//! check: it asserts the comparator *reports* a discrepancy, so the detection path
//! is exercised on every run rather than only during sabotage. Recorded in
//! `docs/verification-log.md`.

mod common;

use common::{GRADIENT_TOLERANCE, STEP, central_difference_gradient, max_relative_discrepancy};

/// `f(x, y) = 3x^2 + 2xy + 5y^2 + 7x - 4y + 11`
fn quadratic(v: &[f64]) -> f64 {
    let (x, y) = (v[0], v[1]);
    3.0 * x * x + 2.0 * x * y + 5.0 * y * y + 7.0 * x - 4.0 * y + 11.0
}

/// `grad f = (6x + 2y + 7, 2x + 10y - 4)`, by hand.
fn quadratic_gradient(v: &[f64]) -> Vec<f64> {
    let (x, y) = (v[0], v[1]);
    vec![6.0 * x + 2.0 * y + 7.0, 2.0 * x + 10.0 * y - 4.0]
}

/// `f(x, y, z) = exp(x) + y*ln(z)`, chosen because its third derivatives are
/// non-zero, so it exercises the truncation error a quadratic cannot.
fn transcendental(v: &[f64]) -> f64 {
    let (x, y, z) = (v[0], v[1], v[2]);
    x.exp() + y * z.ln()
}

/// `grad f = (exp(x), ln(z), y/z)`, by hand.
fn transcendental_gradient(v: &[f64]) -> Vec<f64> {
    let (x, y, z) = (v[0], v[1], v[2]);
    vec![x.exp(), z.ln(), y / z]
}

#[test]
fn central_difference_matches_quadratic() {
    // A quadratic has vanishing third derivative, so the central difference is
    // exact up to round-off. This isolates the harness's arithmetic from its
    // truncation error.
    for point in [
        vec![0.0, 0.0],
        vec![1.0, -1.0],
        vec![-2.5, 3.25],
        vec![100.0, 0.5],
    ] {
        let analytic = quadratic_gradient(&point);
        let numeric = central_difference_gradient(quadratic, &point, STEP);
        let discrepancy = max_relative_discrepancy(&analytic, &numeric);
        assert!(
            discrepancy <= GRADIENT_TOLERANCE,
            "at {point:?}: analytic {analytic:?} vs numeric {numeric:?}, \
             max discrepancy {discrepancy:?} > {GRADIENT_TOLERANCE:?}"
        );
    }
}

#[test]
fn central_difference_matches_transcendental() {
    for point in [
        vec![0.0, 1.0, 1.0],
        vec![0.5, -2.0, 3.0],
        vec![-1.5, 2.0, 0.75],
    ] {
        let analytic = transcendental_gradient(&point);
        let numeric = central_difference_gradient(transcendental, &point, STEP);
        let discrepancy = max_relative_discrepancy(&analytic, &numeric);
        assert!(
            discrepancy <= GRADIENT_TOLERANCE,
            "at {point:?}: analytic {analytic:?} vs numeric {numeric:?}, \
             max discrepancy {discrepancy:?} > {GRADIENT_TOLERANCE:?}"
        );
    }
}

/// The detection path, exercised on every run.
///
/// An oracle that has only ever agreed is not known to be able to disagree. Each
/// case below is a failure mode this harness exists to catch.
#[test]
fn detects_a_wrong_gradient() {
    let point = vec![1.5, -0.75];
    let numeric = central_difference_gradient(quadratic, &point, STEP);
    let correct = quadratic_gradient(&point);

    // A sign error on one component.
    let mut sign_flipped = correct.clone();
    sign_flipped[1] = -sign_flipped[1];
    assert!(
        max_relative_discrepancy(&sign_flipped, &numeric) > GRADIENT_TOLERANCE,
        "harness failed to detect a flipped sign"
    );

    // Two components transposed — the multivariate index error CLAUDE.md §1.3 warns
    // about, which produces plausible-looking numbers.
    let transposed = vec![correct[1], correct[0]];
    assert!(
        max_relative_discrepancy(&transposed, &numeric) > GRADIENT_TOLERANCE,
        "harness failed to detect transposed components"
    );

    // A dropped term: the constant 7 missing from d/dx.
    let missing_term = vec![correct[0] - 7.0, correct[1]];
    assert!(
        max_relative_discrepancy(&missing_term, &numeric) > GRADIENT_TOLERANCE,
        "harness failed to detect a dropped term"
    );

    // A perturbation far too small to notice by eye, but 100x the tolerance.
    let barely_wrong = vec![correct[0] * (1.0 + 1e-5), correct[1]];
    assert!(
        max_relative_discrepancy(&barely_wrong, &numeric) > GRADIENT_TOLERANCE,
        "harness failed to detect a relative perturbation of 1e-5"
    );
}

// Guards the tolerance itself, at compile time.
//
// A runtime test could not do better: both bounds are known statically, so a
// regression should refuse to build rather than wait to be run. Messages must be
// literals in const context, so the reasoning stays here:
//
// - Above 1e-5 the tolerance would admit real derivative errors — a dropped
//   constant term or a mis-scaled factor can easily be that small in relative
//   terms.
// - At or below 1e-8 it would sit under the central difference's own relative
//   round-off floor, observed at 1.07e-9 for `quadratic` at `(100, 0.5)`, and would
//   produce false failures.
const _: () = assert!(
    GRADIENT_TOLERANCE < 1e-5,
    "GRADIENT_TOLERANCE is loose enough to admit real derivative errors"
);
const _: () = assert!(
    GRADIENT_TOLERANCE > 1e-8,
    "GRADIENT_TOLERANCE is below the central difference's own round-off floor"
);
