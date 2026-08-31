//! Shared test-only reference code.
//!
//! Included by several integration-test binaries, each of which uses a different
//! subset, so unused-item warnings here are expected rather than informative.
#![allow(dead_code)]
// Index loops in the multivariate references, for the reason given at the top of
// `hawkes/src/multivariate.rs`: these are transcriptions of indexed formulae and the
// indices are what a reader checks.
#![allow(clippy::needless_range_loop)]

use hawkes::univariate::{Observation, Parameters};

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

/// Agreement required between the `O(n)` recursion and the `O(n^2)` definition.
///
/// Lives here rather than in `loglikelihood.rs` so that `gate_sensitivity.rs` guards
/// the same constant the gate actually uses. A meta-test that pinned its own copy
/// would not notice this one being loosened.
///
/// Justified by measurement in `docs/derivations/check_summation_scaling.py`: the
/// worst observed disagreement is `1.3e-14` relative for `n = 20000` with a slowly
/// decaying kernel, about 75x inside this gate.
pub const RECURSION_TOLERANCE: f64 = 1e-12;

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

/// Asymptotic standard errors of the maximum-likelihood estimate, from the observed
/// Fisher information.
///
/// Standard MLE theory: `Var(theta_hat) ~= I(theta_hat)^-1`, where `I` is the Hessian
/// of the negative log-likelihood at the optimum. That gives a **per-realization**
/// predicted spread derived from theory, rather than from watching this estimator's
/// own scatter — which would be circular, since a biased estimator has a perfectly
/// respectable variance about its own wrong centre.
///
/// The Hessian is obtained by central-differencing the *analytic* gradient, which is
/// itself gated against central differences of the likelihood in `gradient.rs`.
///
/// Returns `None` when the information matrix is singular or the result is not
/// finite — for instance when `beta` is barely identified because `alpha` is small.
pub fn asymptotic_standard_errors(
    parameters: &Parameters,
    observation: &Observation,
) -> Option<[f64; 3]> {
    let centre = [
        parameters.baseline(),
        parameters.excitation(),
        parameters.decay(),
    ];
    // Relative step: the parameters differ in scale, so a fixed absolute step would
    // be badly sized for at least one of them.
    let step: Vec<f64> = centre.iter().map(|v| 1e-5 * v.max(1e-3)).collect();

    let gradient_at = |point: &[f64]| -> [f64; 3] {
        let p = Parameters::new(point[0], point[1], point[2]).expect("valid parameters");
        let (_, g) = hawkes::univariate::negative_log_likelihood_and_gradient(&p, observation);
        [g.baseline, g.excitation, g.decay]
    };

    let mut hessian = [[0.0f64; 3]; 3];
    for j in 0..3 {
        let mut up = centre;
        let mut down = centre;
        up[j] += step[j];
        down[j] -= step[j];
        if down[j] <= 0.0 {
            return None;
        }
        let g_up = gradient_at(&up);
        let g_down = gradient_at(&down);
        for i in 0..3 {
            hessian[i][j] = (g_up[i] - g_down[i]) / (2.0 * step[j]);
        }
    }
    // Symmetrize: the two mixed partials are equal in exact arithmetic and differ
    // only by round-off here.
    //
    // Index loops rather than iterators: this is matrix index arithmetic, where
    // `hessian[i][j]` against `hessian[j][i]` is the point, and clippy's iterator
    // rewrite would hide exactly the thing a reader needs to check.
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in (i + 1)..3 {
            let mean = 0.5 * (hessian[i][j] + hessian[j][i]);
            hessian[i][j] = mean;
            hessian[j][i] = mean;
        }
    }

    let inverse = invert_3x3(&hessian)?;
    let mut errors = [0.0f64; 3];
    for i in 0..3 {
        let variance = inverse[i][i];
        if !variance.is_finite() || variance <= 0.0 {
            return None;
        }
        errors[i] = variance.sqrt();
    }
    Some(errors)
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !determinant.is_finite() || determinant.abs() < 1e-300 {
        return None;
    }
    let mut inverse = [[0.0f64; 3]; 3];
    // Index loops for the same reason as above: the cofactor construction is defined
    // by its indices.
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            // Cofactor of (j, i), giving the transpose of the cofactor matrix.
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            inverse[i][j] = (m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]) / determinant;
        }
    }
    Some(inverse)
}

// ---------------------------------------------------------------- multivariate

use hawkes::multivariate::{Observation as MultiObservation, Parameters as MultiParameters};

/// Brute-force `O(n^2)` multivariate negative log-likelihood: a direct transcription
/// of `docs/derivations/multivariate_loglikelihood.md` (M3.2).
///
/// ```text
/// nll = sum_i mu[i]*T
///     + sum_i sum_j alpha[i][j] * sum_k ( 1 - exp(-beta*(T - t^j_k)) )
///     - sum_i sum_k log( mu[i] + sum_j sum_{t^j_l < t^i_k} alpha[i][j]*beta*exp(-beta*(t^i_k - t^j_l)) )
/// ```
///
/// Same rules as the univariate reference: a transcription, not an optimisation. No
/// pooled walk, no state recursion, no reuse of accumulators. The inner condition is
/// `t^j_l < t^i_k`, **strict and on times**, which is what makes simultaneous events
/// on different components not excite each other.
///
/// Uses the naive `1.0 - exp(-x)` where the production path uses `exp_m1`, so a bug in
/// that choice cannot cancel on both sides.
pub fn brute_force_multivariate_negative_log_likelihood(
    parameters: &MultiParameters,
    observation: &MultiObservation,
) -> f64 {
    let d = parameters.dimension();
    let beta = parameters.decay();
    let horizon = observation.horizon();
    let events = observation.events();

    let mut total = 0.0;
    for i in 0..d {
        total += parameters.baseline()[i] * horizon;
    }
    for i in 0..d {
        for j in 0..d {
            for &t in &events[j] {
                total += parameters.excitation_at(i, j) * (1.0 - (-beta * (horizon - t)).exp());
            }
        }
    }
    for i in 0..d {
        for &t_k in &events[i] {
            let mut intensity = parameters.baseline()[i];
            for j in 0..d {
                for &t_l in &events[j] {
                    if t_l < t_k {
                        intensity +=
                            parameters.excitation_at(i, j) * beta * (-beta * (t_k - t_l)).exp();
                    }
                }
            }
            total -= intensity.ln();
        }
    }
    total
}

/// The scale of the multivariate computation, for use as a comparison denominator.
///
/// Same construction and same reasoning as [`computation_scale`]: the sum of the
/// magnitudes fed into the accumulators, **not** `|nll|`, which is a difference of
/// large terms and passes through zero.
pub fn multivariate_computation_scale(
    parameters: &MultiParameters,
    observation: &MultiObservation,
) -> f64 {
    let d = parameters.dimension();
    let beta = parameters.decay();
    let horizon = observation.horizon();
    let events = observation.events();

    let mut scale = 0.0;
    for i in 0..d {
        scale += parameters.baseline()[i] * horizon;
    }
    for i in 0..d {
        for j in 0..d {
            for &t in &events[j] {
                scale += parameters.excitation_at(i, j) * (1.0 - (-beta * (horizon - t)).exp());
            }
        }
    }
    for i in 0..d {
        for &t_k in &events[i] {
            let mut intensity = parameters.baseline()[i];
            for j in 0..d {
                for &t_l in &events[j] {
                    if t_l < t_k {
                        intensity +=
                            parameters.excitation_at(i, j) * beta * (-beta * (t_k - t_l)).exp();
                    }
                }
            }
            scale += intensity.ln().abs();
        }
    }
    scale
}

/// Deterministic pseudo-random multivariate parameters and events, for tests that
/// must not depend on the simulator.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_f64() * bound as f64) as usize % bound.max(1)
    }
}

/// Asymptotic standard errors for the multivariate MLE, from the observed Fisher
/// information, in the flat layout `[baseline (d), excitation (d*d), decay]`.
///
/// Same construction and same reasoning as [`asymptotic_standard_errors`]: the
/// Hessian of the negative log-likelihood at the optimum, obtained by central-
/// differencing the analytic gradient, inverted. Derived from theory rather than from
/// this estimator's own scatter, which would be circular.
///
/// Returns `None` if the information matrix is singular or any variance is not
/// positive — which happens when a component has too few events to identify its own
/// row of `alpha`.
pub fn multivariate_asymptotic_standard_errors(
    parameters: &MultiParameters,
    observation: &MultiObservation,
) -> Option<Vec<f64>> {
    let d = parameters.dimension();
    let n = d + d * d + 1;
    let mut centre = parameters.baseline().to_vec();
    centre.extend_from_slice(parameters.excitation());
    centre.push(parameters.decay());

    let gradient_at = |point: &[f64]| -> Vec<f64> {
        let p = MultiParameters::new(
            point[..d].to_vec(),
            point[d..d + d * d].to_vec(),
            point[d + d * d],
        )
        .expect("valid parameters");
        let (_, g) =
            hawkes::multivariate::negative_log_likelihood_and_gradient(&p, observation).unwrap();
        let mut flat = g.baseline;
        flat.extend_from_slice(&g.excitation);
        flat.push(g.decay);
        flat
    };

    let mut hessian = vec![0.0f64; n * n];
    for column in 0..n {
        // Clamped to half the coordinate, so a near-zero entry -- which is exactly
        // what a recovered true zero looks like -- is not perturbed below its own
        // domain. Without the clamp the step for a fitted `alpha` of 1e-9 would be
        // 1e-8 and the backward point negative.
        let step = (1e-5 * centre[column].abs().max(1e-3)).min(0.5 * centre[column].abs());
        if step <= 0.0 || !step.is_finite() {
            return None;
        }
        let mut up = centre.clone();
        let mut down = centre.clone();
        up[column] += step;
        down[column] -= step;
        if down[column] <= 0.0 {
            return None;
        }
        let g_up = gradient_at(&up);
        let g_down = gradient_at(&down);
        for row in 0..n {
            hessian[row * n + column] = (g_up[row] - g_down[row]) / (2.0 * step);
        }
    }
    for row in 0..n {
        for column in (row + 1)..n {
            let mean = 0.5 * (hessian[row * n + column] + hessian[column * n + row]);
            hessian[row * n + column] = mean;
            hessian[column * n + row] = mean;
        }
    }

    let inverse = invert(&hessian, n)?;
    let mut errors = Vec::with_capacity(n);
    for i in 0..n {
        let variance = inverse[i * n + i];
        if !variance.is_finite() || variance <= 0.0 {
            return None;
        }
        errors.push(variance.sqrt());
    }
    Some(errors)
}

/// Gauss-Jordan inverse with partial pivoting, row-major `n x n`.
fn invert(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut augmented = vec![0.0f64; n * 2 * n];
    for row in 0..n {
        for column in 0..n {
            augmented[row * 2 * n + column] = matrix[row * n + column];
        }
        augmented[row * 2 * n + n + row] = 1.0;
    }
    for column in 0..n {
        let pivot_row = (column..n).max_by(|&a, &b| {
            augmented[a * 2 * n + column]
                .abs()
                .partial_cmp(&augmented[b * 2 * n + column].abs())
                .unwrap()
        })?;
        if augmented[pivot_row * 2 * n + column].abs() < 1e-300 {
            return None;
        }
        for k in 0..2 * n {
            augmented.swap(column * 2 * n + k, pivot_row * 2 * n + k);
        }
        let pivot = augmented[column * 2 * n + column];
        for k in 0..2 * n {
            augmented[column * 2 * n + k] /= pivot;
        }
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = augmented[row * 2 * n + column];
            if factor == 0.0 {
                continue;
            }
            for k in 0..2 * n {
                augmented[row * 2 * n + k] -= factor * augmented[column * 2 * n + k];
            }
        }
    }
    let mut inverse = vec![0.0f64; n * n];
    for row in 0..n {
        for column in 0..n {
            inverse[row * n + column] = augmented[row * 2 * n + n + column];
        }
    }
    Some(inverse)
}
