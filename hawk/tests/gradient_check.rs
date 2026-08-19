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

/// Step size for the central difference.
///
/// Central differences carry truncation error `O(h^2 * f''')` and round-off error
/// `O(eps_machine / h)`, so the total is minimised near `h = eps^(1/3) ~ 6e-6` for
/// f64. 1e-5 sits just above that, trading a little truncation error for margin
/// against round-off on functions whose scale is larger than unity.
const STEP: f64 = 1e-5;

/// Agreement required between an analytic gradient and the central difference,
/// measured **relatively** (see [`max_relative_discrepancy`]).
///
/// The measure has to be relative. The round-off floor of a central difference is
/// `eps * |f| / (h * |f'|)`, which grows with the *value* of `f`, not with its
/// derivative. An absolute tolerance therefore cannot hold uniformly: at
/// `(x, y) = (100, 0.5)` the quadratic below has `|f| ~ 3e4` and the observed
/// absolute error is `6.5e-7`, while at the origin it is around `1e-11`. The first
/// version of this harness used an absolute tolerance and failed on exactly that
/// point.
///
/// Relatively, that same worst case is `1.07e-9`, matching the predicted floor. A
/// tolerance of 1e-7 leaves roughly two orders of magnitude of headroom over the
/// floor while still catching a sign error, a transposed index, a dropped term, or
/// a relative perturbation of 1e-5 — the failures this oracle exists for.
const GRADIENT_TOLERANCE: f64 = 1e-7;

/// Central-difference approximation to the gradient of `f` at `point`.
///
/// `(f(x + h e_i) - f(x - h e_i)) / 2h` per coordinate. Central rather than forward
/// differences because the error is `O(h^2)` rather than `O(h)`, which is what makes
/// a tolerance tight enough to catch a real derivative bug achievable at all.
fn central_difference_gradient<F>(f: F, point: &[f64], step: f64) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut gradient = Vec::with_capacity(point.len());
    let mut perturbed = point.to_vec();

    for index in 0..point.len() {
        let original = point[index];

        perturbed[index] = original + step;
        let forward = f(&perturbed);

        perturbed[index] = original - step;
        let backward = f(&perturbed);

        perturbed[index] = original;
        gradient.push((forward - backward) / (2.0 * step));
    }

    gradient
}

/// Largest disagreement between two gradients, relative to their own magnitude.
///
/// Each component is scaled by `max(1, |analytic|, |numeric|)`. The floor of 1
/// keeps the measure absolute for components near zero, where a relative error is
/// meaningless and would otherwise divide by almost nothing.
///
/// Returned rather than asserted so that both outcomes are testable: a correct
/// gradient must produce a small value, and a wrong one must produce a large value.
fn max_relative_discrepancy(analytic: &[f64], numeric: &[f64]) -> f64 {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "gradients have different lengths: {} and {}",
        analytic.len(),
        numeric.len()
    );
    analytic
        .iter()
        .zip(numeric)
        .map(|(a, n)| (a - n).abs() / a.abs().max(n.abs()).max(1.0))
        .fold(0.0f64, f64::max)
}

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
