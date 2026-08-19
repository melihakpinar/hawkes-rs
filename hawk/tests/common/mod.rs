//! Shared test-only reference code.
//!
//! Included by several integration-test binaries, each of which uses a different
//! subset, so unused-item warnings here are expected rather than informative.
#![allow(dead_code)]

use hawk::univariate::{Observation, Parameters};

/// Brute-force `O(n^2)` negative log-likelihood: a direct transcription of the
/// **definition**, `docs/derivations/univariate_loglikelihood.md` (3.3).
///
/// ```text
/// nll = mu*T
///     + alpha * sum_{i=1}^{n} ( 1 - exp(-beta*(T - t_i)) )
///     - sum_{k=1}^{n} log( mu + sum_{i : t_i < t_k} alpha*beta*exp(-beta*(t_k - t_i)) )
/// ```
///
/// This exists to be *obviously* correct rather than fast, so it must stay a
/// transcription. No recursion, no algebraic simplification, no accumulator reuse —
/// every deviation is a chance for it to share a bug with the implementation it is
/// supposed to check.
///
/// Deliberately uses `1.0 - (-x).exp()` rather than `-(-x).exp_m1()`. The production
/// path uses `exp_m1` (§5's numerical notes), and if the reference used it too a bug
/// in that choice would cancel on both sides and be invisible. The naive form's
/// cancellation costs at most one `f64` epsilon of *absolute* error per term — the
/// subtraction itself is exact by Sterbenz whenever `exp(-x)` is near 1 — which is
/// far inside the gate, since the gate is relative to the scale of the computation
/// rather than to `nll`.
///
/// The inner sum uses `t_i < t_k` strictly, on **times** rather than on indices, so
/// simultaneous events do not excite each other (`conventions.md` C3, C8).
pub fn brute_force_negative_log_likelihood(
    parameters: &Parameters,
    observation: &Observation,
) -> f64 {
    let mu = parameters.baseline();
    let alpha = parameters.excitation();
    let beta = parameters.decay();
    let horizon = observation.horizon();
    let times = observation.times();

    let mut compensator = mu * horizon;
    for &t_i in times {
        compensator += alpha * (1.0 - (-beta * (horizon - t_i)).exp());
    }

    let mut log_term = 0.0;
    for &t_k in times {
        let mut intensity = mu;
        for &t_i in times {
            if t_i < t_k {
                intensity += alpha * beta * (-beta * (t_k - t_i)).exp();
            }
        }
        log_term += intensity.ln();
    }

    compensator - log_term
}

/// The scale of the computation, used as the denominator when comparing two
/// implementations. See `univariate_loglikelihood.md` §5.
///
/// **Not** `|nll|`. `nll = mu*T + alpha*compensator - log_term` is a difference of
/// large terms and passes through zero, so at parameter points where those terms
/// cancel a relative error taken against `|nll|` diverges on *correct* code. A
/// randomized sweep reaches that region — it is a surface through the parameter
/// space, not a corner of it.
///
/// Floating-point error is proportional to the magnitudes fed into the accumulators,
/// not to how much they happen to cancel at the end, which is what this measures:
///
/// ```text
/// scale = mu*T + alpha*sum_i (1 - exp(-beta*(T - t_i))) + sum_k |log lambda(t_k)|
/// ```
///
/// The last term sums **magnitudes**. `|sum_k log lambda(t_k)|` would carry the same
/// defect one level down: `log lambda` is negative wherever `lambda < 1`, routine for
/// `mu < 1`, so a near-zero `log_term` can be thousands of `O(1)` contributions
/// cancelling.
pub fn computation_scale(parameters: &Parameters, observation: &Observation) -> f64 {
    let mu = parameters.baseline();
    let alpha = parameters.excitation();
    let beta = parameters.decay();
    let horizon = observation.horizon();
    let times = observation.times();

    let mut scale = mu * horizon;
    for &t_i in times {
        scale += alpha * (1.0 - (-beta * (horizon - t_i)).exp());
    }
    for &t_k in times {
        let mut intensity = mu;
        for &t_i in times {
            if t_i < t_k {
                intensity += alpha * beta * (-beta * (t_k - t_i)).exp();
            }
        }
        scale += intensity.ln().abs();
    }
    scale
}

/// Step size for the central difference.
///
/// Central differences carry truncation error `O(h^2 * f''')` and round-off error
/// `O(eps_machine / h)`, so the total is minimised near `h = eps^(1/3) ~ 6e-6` for
/// f64. 1e-5 sits just above that, trading a little truncation error for margin
/// against round-off on functions whose scale is larger than unity.
pub const STEP: f64 = 1e-5;

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
pub const GRADIENT_TOLERANCE: f64 = 1e-7;

/// Central-difference approximation to the gradient of `f` at `point`.
///
/// `(f(x + h e_i) - f(x - h e_i)) / 2h` per coordinate. Central rather than forward
/// differences because the error is `O(h^2)` rather than `O(h)`, which is what makes
/// a tolerance tight enough to catch a real derivative bug achievable at all.
pub fn central_difference_gradient<F>(f: F, point: &[f64], step: f64) -> Vec<f64>
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
pub fn max_relative_discrepancy(analytic: &[f64], numeric: &[f64]) -> f64 {
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
