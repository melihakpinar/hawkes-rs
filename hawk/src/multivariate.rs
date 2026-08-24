//! Multivariate exponential-kernel Hawkes process.
//!
//! Transcribed from `docs/derivations/multivariate_loglikelihood.md` and
//! `docs/derivations/multivariate_gradient.md`. Equation references of the form
//! `(M4.4)` and `(MG.8)` point there; `(4.4)`, `(G.6)` point at the univariate pair.
//!
//! # Orientation
//!
//! `excitation[i][j]` means **"j excites i"** (`conventions.md` C6, confirmed
//! independently by [Laub2015, eq. 13]). The first index is the component being
//! excited. A transposed matrix produces plausible numbers and is only detectable on
//! asymmetric data, which is why the corpus carries asymmetric fixtures and the
//! round-trip test compares elementwise.

// Index loops over `0..d` are used throughout rather than iterators.
//
// This module is matrix arithmetic, and the indices are the content: `excitation[i*d+j]`
// against `state[j]`, `pair_accumulator[i*d+j]` against `decay_compensator[j]`. Which
// index is the excited component and which the exciting one is exactly what a reader
// has to check against `conventions.md` C6, and clippy's iterator rewrite hides it
// behind a zip whose operands are no longer named. The lint is right in general and
// wrong here.
#![allow(clippy::needless_range_loop)]

use crate::Error;

/// Parameters of a `d`-component exponential-kernel Hawkes process.
///
/// The intensity is
///
/// ```text
/// lambda_i(t) = mu[i] + sum_j sum_{t^j_k < t} alpha[i][j] * beta * exp(-beta*(t - t^j_k))
/// ```
///
/// with the sums strictly over earlier events, so `lambda` is predictable
/// (`conventions.md` C3).
#[derive(Debug, Clone, PartialEq)]
pub struct Parameters {
    /// `mu`, one per component.
    baseline: Vec<f64>,
    /// `alpha`, row-major `d x d`. `excitation[i*d + j]` is "j excites i".
    excitation: Vec<f64>,
    /// `beta`, shared by every pair.
    ///
    /// A per-pair decay matrix is out of scope: `tick`'s
    /// `ModelHawkesExpKernLogLik` takes a scalar and is the differential oracle, so
    /// nothing available could check one (`multivariate_loglikelihood.md` §0.1).
    decay: f64,
}

impl Parameters {
    /// Validates and constructs from a row-major excitation matrix.
    ///
    /// `baseline` and `decay` must be strictly positive; excitation entries must be
    /// non-negative. **Zero excitation entries are legitimate** — a component that
    /// excites nothing, or is excited by nothing, is ordinary in `d` dimensions,
    /// unlike the univariate case where `alpha = 0` is a degenerate Poisson process.
    ///
    /// Stationarity is deliberately not checked here (CLAUDE.md §6); see
    /// [`Parameters::is_stationary`].
    pub fn new(baseline: Vec<f64>, excitation: Vec<f64>, decay: f64) -> Result<Self, Error> {
        let dimension = baseline.len();
        if dimension == 0 {
            return Err(Error::EmptyProcess);
        }
        if excitation.len() != dimension * dimension {
            return Err(Error::DimensionMismatch {
                what: "excitation",
                actual: excitation.len(),
                expected: dimension * dimension,
                dimension,
            });
        }
        for (index, &value) in baseline.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::NonFiniteParameter {
                    name: "baseline",
                    value,
                });
            }
            if value <= 0.0 {
                let _ = index;
                return Err(Error::NonPositiveParameter {
                    name: "baseline",
                    value,
                });
            }
        }
        for (index, &value) in excitation.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::InvalidExcitation {
                    row: index / dimension,
                    column: index % dimension,
                    value,
                });
            }
        }
        if !decay.is_finite() {
            return Err(Error::NonFiniteParameter {
                name: "decay",
                value: decay,
            });
        }
        if decay <= 0.0 {
            return Err(Error::NonPositiveParameter {
                name: "decay",
                value: decay,
            });
        }
        Ok(Self {
            baseline,
            excitation,
            decay,
        })
    }

    pub fn dimension(&self) -> usize {
        self.baseline.len()
    }

    pub fn baseline(&self) -> &[f64] {
        &self.baseline
    }

    /// Row-major `d x d`; index as `excitation()[i * dimension() + j]`.
    pub fn excitation(&self) -> &[f64] {
        &self.excitation
    }

    /// `alpha[i][j]`, i.e. how much `j` excites `i`.
    pub fn excitation_at(&self, excited: usize, exciting: usize) -> f64 {
        self.excitation[excited * self.dimension() + exciting]
    }

    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Spectral radius of the branching matrix `alpha`.
    ///
    /// Under this crate's kernel normalization the kernel `(i, j)` integrates to
    /// `alpha[i][j]` (`conventions.md` C1, C2), so `alpha` *is* the matrix of kernel
    /// `L1` norms and its spectral radius is the quantity
    /// [Bacry2015, Proposition 1] requires to be below 1.
    ///
    /// # Method
    ///
    /// `alpha` is non-negative, so its spectral radius is its Perron root and can be
    /// bracketed by the Collatz-Wielandt bounds: for any strictly positive `x`,
    ///
    /// ```text
    /// min_i (A x)_i / x_i  <=  rho(A)  <=  max_i (A x)_i / x_i
    /// ```
    ///
    /// The **upper** bound is what is returned, not the bracket's midpoint. Collatz-
    /// Wielandt gives `rho(A) = inf over positive x of max_i (A x)_i / x_i`, so the
    /// upper bound converges to `rho` from above for any non-negative matrix,
    /// including reducible ones. The lower bound converges only when `A` is
    /// irreducible: for a diagonal matrix `(A x)_i / x_i = m_i` exactly, so the
    /// bracket is `[min m_i, max m_i]` at every step and never closes. Returning the
    /// midpoint gives `0.45` for `diag(0.2, 0.7, 0.4)`, whose spectral radius is
    /// `0.7` — that case is in the tests.
    ///
    /// Power iteration is run on `A + I` rather than `A`, and 1 subtracted at the
    /// end: `rho(A + I) = rho(A) + 1` for non-negative `A`, and shifting makes the
    /// matrix aperiodic. Without the shift a periodic matrix such as
    /// `[[0, 2], [0.9, 0]]` oscillates forever between the bounds `0.9` and `2` and
    /// never converges — also in the tests.
    ///
    /// # Accuracy
    ///
    /// Exact to `1e-9` or better for diagonalizable matrices, which is every case the
    /// tests pin by hand.
    ///
    /// **Defective** matrices — a repeated eigenvalue with too few eigenvectors, such
    /// as `[[0.4, 7], [0, 0.4]]` — converge sublinearly, like `1/k` rather than
    /// geometrically, and land within about `3e-4` at the iteration cap. That is
    /// immaterial for the use this value has: deciding `rho < 1`, and reporting a
    /// diagnostic. It would matter to a caller reading it as a precise quantity, so it
    /// is stated rather than left to be discovered. `spectral_radius.rs` pins the
    /// defective cases at that looser tolerance and says why.
    pub fn branching_ratio_spectral_radius(&self) -> f64 {
        let d = self.dimension();
        let shifted =
            |i: usize, j: usize| self.excitation[i * d + j] + if i == j { 1.0 } else { 0.0 };

        let mut x = vec![1.0f64; d];
        let mut lower = 0.0f64;
        let mut upper = f64::INFINITY;

        // Geometric convergence; the cap is a backstop, not the expected exit.
        for _ in 0..10_000 {
            let y: Vec<f64> = (0..d)
                .map(|i| {
                    let mut acc = 0.0;
                    for j in 0..d {
                        acc += shifted(i, j) * x[j];
                    }
                    acc
                })
                .collect();
            lower = f64::INFINITY;
            upper = 0.0;
            for i in 0..d {
                let ratio = y[i] / x[i];
                lower = lower.min(ratio);
                upper = upper.max(ratio);
            }
            // Both bounds are rigorous at every step, so stopping when the bracket
            // closes is safe. There is deliberately NO "the upper bound stopped
            // moving" exit: the upper bound can plateau for a few iterations and then
            // resume falling. On the nilpotent matrix `[[0,1,0],[0,0,1],[0,0,0]]` it
            // sits at 2 for two steps before descending towards 1, and exiting there
            // returned a spectral radius of 1.0 for a matrix whose radius is 0 — a
            // trivially stationary process reported as explosive.
            if upper - lower <= 1e-14 * upper.max(1.0) {
                break;
            }
            let norm: f64 = y.iter().sum();
            if norm == 0.0 || !norm.is_finite() {
                break;
            }
            for (slot, value) in x.iter_mut().zip(&y) {
                *slot = value / norm * d as f64;
            }
        }
        let _ = lower;
        // The upper bound, shifted back. See the note above on why not the midpoint.
        (upper - 1.0).max(0.0)
    }

    /// Whether the process is stationary: branching-ratio spectral radius `< 1`.
    ///
    /// Reported as a diagnostic, never enforced (CLAUDE.md §6).
    pub fn is_stationary(&self) -> bool {
        self.branching_ratio_spectral_radius() < 1.0
    }

    /// Stationary mean intensity `(I - alpha)^{-1} mu`, or `None` when the process is
    /// not stationary.
    ///
    /// `multivariate_loglikelihood.md` (M7.1), from [Bacry2015, Proposition 4,
    /// eq. 21]. At `d = 1` this is `mu / (1 - alpha)`, [Laub2015, eq. 6].
    ///
    /// # The spectral radius is checked, not inferred from the solve
    ///
    /// Invertibility of `I - alpha` does **not** establish stationarity. The matrix
    /// `alpha = [[0, 2], [0.9, 0]]` has spectral radius `1.34` and `I - alpha` has
    /// determinant `-0.8`, so the solve succeeds and returns a vector with negative
    /// entries which is not a mean intensity of anything. Returning it would be worse
    /// than failing, because it looks like an answer.
    ///
    /// So (M7.2) is tested first and independently. A successful solve is not taken
    /// as evidence of anything.
    pub fn stationary_mean_intensity(&self) -> Option<Vec<f64>> {
        if !self.is_stationary() {
            return None;
        }
        let d = self.dimension();
        // Augmented [I - alpha | mu], solved by Gauss-Jordan with partial pivoting.
        let mut augmented = vec![0.0f64; d * (d + 1)];
        for i in 0..d {
            for j in 0..d {
                augmented[i * (d + 1) + j] =
                    if i == j { 1.0 } else { 0.0 } - self.excitation[i * d + j];
            }
            augmented[i * (d + 1) + d] = self.baseline[i];
        }
        for column in 0..d {
            let pivot_row = (column..d)
                .max_by(|&a, &b| {
                    augmented[a * (d + 1) + column]
                        .abs()
                        .partial_cmp(&augmented[b * (d + 1) + column].abs())
                        .expect("matrix entries are finite")
                })
                .expect("at least one row remains");
            if augmented[pivot_row * (d + 1) + column] == 0.0 {
                // Singular. Unreachable when the spectral radius is below 1, but
                // returning None is the honest response rather than dividing by zero.
                return None;
            }
            for k in 0..=d {
                augmented.swap(column * (d + 1) + k, pivot_row * (d + 1) + k);
            }
            let pivot = augmented[column * (d + 1) + column];
            for k in column..=d {
                augmented[column * (d + 1) + k] /= pivot;
            }
            for row in 0..d {
                if row == column {
                    continue;
                }
                let factor = augmented[row * (d + 1) + column];
                if factor == 0.0 {
                    continue;
                }
                for k in column..=d {
                    augmented[row * (d + 1) + k] -= factor * augmented[column * (d + 1) + k];
                }
            }
        }
        Some((0..d).map(|i| augmented[i * (d + 1) + d]).collect())
    }
}

/// A `d`-component realization observed on `[0, horizon]`, validated against the
/// input contract (`conventions.md` C8).
#[derive(Debug, Clone, Copy)]
pub struct Observation<'a> {
    events: &'a [Vec<f64>],
    horizon: f64,
}

impl<'a> Observation<'a> {
    /// Validates and constructs. `events[j]` holds component `j`'s timestamps,
    /// ascending, within `[0, horizon]`; ties are permitted, within a component and
    /// across components.
    ///
    /// Cross-component order is not defined and not required (C8): the components are
    /// separate sequences and nothing depends on how they interleave in storage.
    pub fn new(events: &'a [Vec<f64>], horizon: f64) -> Result<Self, Error> {
        if events.is_empty() {
            return Err(Error::EmptyProcess);
        }
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(Error::InvalidHorizon { horizon });
        }
        for component in events {
            for (index, &time) in component.iter().enumerate() {
                if !time.is_finite() {
                    return Err(Error::NonFiniteEvent { index });
                }
                if time < 0.0 || time > horizon {
                    return Err(Error::EventOutsideWindow {
                        index,
                        time,
                        horizon,
                    });
                }
                if index > 0 && time < component[index - 1] {
                    return Err(Error::UnsortedEvents {
                        index,
                        previous_index: index - 1,
                        previous: component[index - 1],
                        current: time,
                    });
                }
            }
        }
        Ok(Self { events, horizon })
    }

    pub fn dimension(&self) -> usize {
        self.events.len()
    }

    pub fn events(&self) -> &'a [Vec<f64>] {
        self.events
    }

    pub fn horizon(&self) -> f64 {
        self.horizon
    }

    /// Total number of events across all components.
    pub fn len(&self) -> usize {
        self.events.iter().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.events.iter().all(Vec::is_empty)
    }
}

/// Partial derivatives of the negative log-likelihood.
///
/// From `multivariate_gradient.md`, equations (MG.1), (MG.2) and (MG.8).
#[derive(Debug, Clone, PartialEq)]
pub struct Gradient {
    /// `d nll / d mu[i]` — (MG.1).
    pub baseline: Vec<f64>,
    /// `d nll / d alpha[i][j]`, row-major `d x d` — (MG.2).
    pub excitation: Vec<f64>,
    /// `d nll / d beta` — (MG.8).
    pub decay: f64,
}

impl Gradient {
    /// Converts to log-parameter space by the chain rule, equation (MG.9).
    ///
    /// See `multivariate_gradient.md` §6 and `docs/derivations/parameter_space.md`
    /// for why the *fit* does not use this for `excitation`.
    pub fn to_log_parameter_space(&self, parameters: &Parameters) -> Self {
        Self {
            baseline: self
                .baseline
                .iter()
                .zip(parameters.baseline())
                .map(|(g, p)| p * g)
                .collect(),
            excitation: self
                .excitation
                .iter()
                .zip(parameters.excitation())
                .map(|(g, p)| p * g)
                .collect(),
            decay: parameters.decay() * self.decay,
        }
    }
}

/// Walks the pooled distinct event times across all components.
///
/// Pooling is not an optimisation, it is required: an event of component `j` at
/// exactly time `t` must not contribute to `lambda_i(t)` for any `i`, so every
/// component's state has to be advanced to `t` *before* any of the events at `t` are
/// absorbed. Advancing per event instead lets whichever component the merge visits
/// first excite the others — see `multivariate_loglikelihood.md` §4.1, where that
/// mistake is off by 1.1% on a four-event example.
struct PooledWalk<'a> {
    events: &'a [Vec<f64>],
    cursor: Vec<usize>,
}

impl<'a> PooledWalk<'a> {
    fn new(events: &'a [Vec<f64>]) -> Self {
        Self {
            cursor: vec![0; events.len()],
            events,
        }
    }

    /// The next distinct time and, per component, how many events sit at it.
    fn next(&mut self, counts: &mut [usize]) -> Option<f64> {
        let mut time = f64::INFINITY;
        for (component, &cursor) in self.events.iter().zip(&self.cursor) {
            if let Some(&candidate) = component.get(cursor)
                && candidate < time
            {
                time = candidate;
            }
        }
        if !time.is_finite() {
            return None;
        }
        for (index, component) in self.events.iter().enumerate() {
            counts[index] = 0;
            while component.get(self.cursor[index]) == Some(&time) {
                self.cursor[index] += 1;
                counts[index] += 1;
            }
        }
        Some(time)
    }
}

/// Asserts that a `Parameters` and an `Observation` describe the same process.
///
/// # Panics
///
/// If the dimensions disagree. This is a documented invariant of the pair rather than
/// a data error (CLAUDE.md §5): in Rust the two are separate arguments and a mismatch
/// is a programming mistake, which is how `ndarray` and `nalgebra` treat a shape
/// mismatch too.
///
/// It is not a theoretical hazard. Before this check existed, `d = 3` parameters with
/// two components of events **silently returned a number** — the missing component was
/// treated as empty — and `d = 2` parameters with three components read past the end of
/// the counts array and panicked with `index out of bounds`. Both were found by the
/// Python bindings' error-mapping test, which requires that no panic reach the
/// interpreter.
///
/// A caller taking dimensions from user input must check first and return an error.
/// `hawk-python` does, so a Python caller gets `ValueError`.
#[track_caller]
fn assert_dimensions_agree(parameters: &Parameters, observation: &Observation) {
    assert_eq!(
        parameters.dimension(),
        observation.dimension(),
        "parameters describe a {}-component process but the observation has {} \
         components; they must agree",
        parameters.dimension(),
        observation.dimension()
    );
}

/// Negative log-likelihood, `O(n*d)`.
///
/// Transcription of `multivariate_loglikelihood.md` (M4.6) and §5.
///
/// # Bit-identical with the gradient path, and with `univariate` at `d = 1`
///
/// Two accumulation rules make that hold, both compatibility constraints rather than
/// numerical improvements (`multivariate_loglikelihood.md` §5.1):
///
/// - Accumulate **per event**, not `count * value`. For a tie of multiplicity 7 the
///   two differ in `f64`.
/// - Group the intensity product as `(alpha[i][j] * beta) * state[j]`, because
///   floating-point multiplication is not associative and `univariate` computes
///   `(alpha*beta)*state`.
pub fn negative_log_likelihood(parameters: &Parameters, observation: &Observation) -> f64 {
    assert_dimensions_agree(parameters, observation);
    let d = parameters.dimension();
    let beta = parameters.decay;
    let horizon = observation.horizon();

    let mut total: f64 = 0.0;
    for i in 0..d {
        total += parameters.baseline[i] * horizon;
    }
    if observation.is_empty() {
        return total;
    }

    let mut walk = PooledWalk::new(observation.events());
    let mut counts = vec![0usize; d];
    let mut state = vec![0.0f64; d];
    let mut compensator = vec![0.0f64; d];
    // Accumulated per component and combined in index order at the end, so the
    // summation order does not depend on how the per-component work is scheduled.
    // That is what lets the parallel path be bitwise identical (step 14); it is not
    // an optimisation.
    let mut log_term_parts = vec![0.0f64; d];
    let mut previous_time = 0.0f64;
    let mut previous_counts = vec![0usize; d];
    let mut started = false;

    while let Some(time) = walk.next(&mut counts) {
        if started {
            let gap = time - previous_time;
            for m in 0..d {
                let (advanced, _) = crate::univariate::advance_excitation_state(
                    state[m],
                    previous_counts[m] as f64,
                    gap,
                    beta,
                );
                state[m] = advanced;
            }
        }
        started = true;

        let contribution = crate::univariate::compensator_contribution(beta, horizon - time);
        for j in 0..d {
            for _ in 0..counts[j] {
                compensator[j] += contribution;
            }
        }

        for i in 0..d {
            if counts[i] == 0 {
                continue;
            }
            let mut intensity = parameters.baseline[i];
            for j in 0..d {
                intensity += (parameters.excitation[i * d + j] * beta) * state[j];
            }
            let logarithm = intensity.ln();
            for _ in 0..counts[i] {
                log_term_parts[i] += logarithm;
            }
        }

        previous_time = time;
        previous_counts.copy_from_slice(&counts);
    }

    for i in 0..d {
        for j in 0..d {
            total += parameters.excitation[i * d + j] * compensator[j];
        }
    }
    let mut log_term = 0.0f64;
    for part in &log_term_parts {
        log_term += part;
    }
    total - log_term
}

/// Negative log-likelihood and its analytic gradient, in one `O(n*d)` pass.
///
/// Transcription of `multivariate_gradient.md` §5.
///
/// The `d/dbeta` assembly uses the per-pair accumulator (MG.7) and forms
/// `sum_ij alpha[i][j] * (dE_j - S[i][j])`, equation (MG.8). That factoring is chosen
/// so `d = 1` reduces bitwise to `univariate`'s `alpha * (X - Y)` association; the
/// natural alternative does not, and would have required re-associating already
/// approved M1 code.
pub fn negative_log_likelihood_and_gradient(
    parameters: &Parameters,
    observation: &Observation,
) -> (f64, Gradient) {
    assert_dimensions_agree(parameters, observation);
    let d = parameters.dimension();
    let beta = parameters.decay;
    let horizon = observation.horizon();

    let mut total: f64 = 0.0;
    for i in 0..d {
        total += parameters.baseline[i] * horizon;
    }
    if observation.is_empty() {
        return (
            total,
            Gradient {
                baseline: vec![horizon; d],
                excitation: vec![0.0; d * d],
                decay: 0.0,
            },
        );
    }

    let mut walk = PooledWalk::new(observation.events());
    let mut counts = vec![0usize; d];
    let mut previous_counts = vec![0usize; d];
    let mut state = vec![0.0f64; d];
    let mut state_derivative = vec![0.0f64; d];
    let mut compensator = vec![0.0f64; d];
    let mut decay_compensator = vec![0.0f64; d];
    // Per component, combined in index order at the end. See the note in
    // `negative_log_likelihood`.
    let mut log_term_parts = vec![0.0f64; d];
    let mut baseline_accumulator = vec![0.0f64; d];
    let mut excitation_accumulator = vec![0.0f64; d * d];
    let mut pair_accumulator = vec![0.0f64; d * d]; // S[i][j], (MG.7)
    let mut previous_time = 0.0f64;
    let mut started = false;

    while let Some(time) = walk.next(&mut counts) {
        if started {
            let gap = time - previous_time;
            for m in 0..d {
                let (advanced, gap_decay) = crate::univariate::advance_excitation_state(
                    state[m],
                    previous_counts[m] as f64,
                    gap,
                    beta,
                );
                // (MG.6): uses the advanced value, written before `state[m]` is
                // overwritten. M1's two hazards, once per component.
                state_derivative[m] = -gap * advanced + gap_decay * state_derivative[m];
                state[m] = advanced;
            }
        }
        started = true;

        let window = horizon - time;
        let window_decay = (-beta * window).exp();
        let contribution = crate::univariate::compensator_contribution(beta, window);
        for j in 0..d {
            for _ in 0..counts[j] {
                compensator[j] += contribution;
                decay_compensator[j] += window * window_decay;
            }
        }

        for i in 0..d {
            if counts[i] == 0 {
                continue;
            }
            let mut intensity = parameters.baseline[i];
            for j in 0..d {
                intensity += (parameters.excitation[i * d + j] * beta) * state[j];
            }
            let logarithm = intensity.ln();
            for _ in 0..counts[i] {
                log_term_parts[i] += logarithm;
                baseline_accumulator[i] += 1.0 / intensity;
                for j in 0..d {
                    excitation_accumulator[i * d + j] += (beta * state[j]) / intensity;
                    pair_accumulator[i * d + j] +=
                        (state[j] + beta * state_derivative[j]) / intensity;
                }
            }
        }

        previous_time = time;
        previous_counts.copy_from_slice(&counts);
    }

    for i in 0..d {
        for j in 0..d {
            total += parameters.excitation[i * d + j] * compensator[j];
        }
    }
    let mut log_term = 0.0f64;
    for part in &log_term_parts {
        log_term += part;
    }
    let negative_log_likelihood = total - log_term;

    let baseline = (0..d).map(|i| horizon - baseline_accumulator[i]).collect();
    let mut excitation = vec![0.0f64; d * d];
    let mut decay = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            excitation[i * d + j] = compensator[j] - excitation_accumulator[i * d + j];
            decay += parameters.excitation[i * d + j]
                * (decay_compensator[j] - pair_accumulator[i * d + j]);
        }
    }

    (
        negative_log_likelihood,
        Gradient {
            baseline,
            excitation,
            decay,
        },
    )
}

/// The compensator `Lambda_i(t) = int_0^t lambda_i(u) du`, evaluated at each
/// component's own event times.
///
/// `result[i][k]` is `Lambda_i` at component `i`'s `k`-th event.
///
/// ```text
/// Lambda_i(t) = mu[i]*t + sum_j alpha[i][j] * sum_{t^j_l < t} ( 1 - exp(-beta*(t - t^j_l)) )
/// ```
///
/// The inner sum telescopes to `m_j(t) - B^j(t)`, where `m_j(t)` counts the events of
/// `j` strictly before `t`, so this runs in `O(n*d)` on the same pooled recursion the
/// likelihood uses.
///
/// For Ogata residual analysis: by the random time change theorem
/// [Laub2015, Theorem 4] applied per component, if the parameters are correct then
/// each `{Lambda_i(t^i_k)}_k` is a unit-rate Poisson process, so the successive
/// differences are i.i.d. `Exp(1)`. Note this must be checked **per component**: a
/// pooled test would let an error in one component be masked by another.
pub fn compensator_at_events(parameters: &Parameters, observation: &Observation) -> Vec<Vec<f64>> {
    assert_dimensions_agree(parameters, observation);
    let d = parameters.dimension();
    let beta = parameters.decay;
    let events = observation.events();
    let mut out: Vec<Vec<f64>> = events.iter().map(|c| Vec::with_capacity(c.len())).collect();
    if observation.is_empty() {
        return out;
    }

    let mut walk = PooledWalk::new(events);
    let mut counts = vec![0usize; d];
    let mut previous_counts = vec![0usize; d];
    let mut state = vec![0.0f64; d];
    let mut before = vec![0.0f64; d];
    let mut previous_time = 0.0f64;
    let mut started = false;

    while let Some(time) = walk.next(&mut counts) {
        if started {
            let gap = time - previous_time;
            for m in 0..d {
                let (advanced, _) = crate::univariate::advance_excitation_state(
                    state[m],
                    previous_counts[m] as f64,
                    gap,
                    beta,
                );
                state[m] = advanced;
                before[m] += previous_counts[m] as f64;
            }
        }
        started = true;

        for i in 0..d {
            if counts[i] == 0 {
                continue;
            }
            let mut value = parameters.baseline[i] * time;
            for j in 0..d {
                value += parameters.excitation[i * d + j] * (before[j] - state[j]);
            }
            for _ in 0..counts[i] {
                out[i].push(value);
            }
        }

        previous_time = time;
        previous_counts.copy_from_slice(&counts);
    }
    out
}

/// Simulates a `d`-component realization on `[0, horizon]` by Ogata's modified
/// thinning algorithm, [Laub2015, Algorithm 2] extended to `d` dimensions.
///
/// The bound is the **total** intensity `sum_i lambda_i(t+)`, taken from the exact
/// right-hand limit rather than [Laub2015]'s `lambda*(t + epsilon)` nudge, for the
/// reason given in [`crate::univariate::simulate`]. Between accepted events every
/// component's excitation decays and `alpha` is non-negative, so the total is
/// non-increasing and the bound holds until the next acceptance.
///
/// On acceptance the component is drawn with probability `lambda_i / sum_i lambda_i`,
/// which is the standard superposition argument: the multivariate process is the
/// superposition of `d` point processes, and conditional on a point occurring, it
/// belongs to component `i` with that probability.
///
/// # Non-stationary parameters
///
/// No cap on the number of events. With `spectral_radius(alpha) >= 1`
/// ([`Parameters::is_stationary`]) the process is explosive; check first.
pub fn simulate(
    parameters: &Parameters,
    horizon: f64,
    rng: &mut impl rand::Rng,
) -> Result<Vec<Vec<f64>>, Error> {
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err(Error::InvalidHorizon { horizon });
    }
    let d = parameters.dimension();
    let beta = parameters.decay;

    let mut events: Vec<Vec<f64>> = vec![Vec::new(); d];
    let mut current = 0.0f64;
    // sum_{t^m_k <= current} exp(-beta*(current - t^m_k)); the RIGHT limit, so
    // `intensity_of` below gives lambda_i(current+).
    let mut excitation = vec![0.0f64; d];
    let mut intensity = vec![0.0f64; d];

    loop {
        let mut bound = 0.0;
        for i in 0..d {
            let mut value = parameters.baseline[i];
            for j in 0..d {
                value += (parameters.excitation[i * d + j] * beta) * excitation[j];
            }
            bound += value;
        }

        let uniform: f64 = rng.random::<f64>();
        let waiting = -uniform.ln() / bound;
        let candidate = current + waiting;
        if candidate >= horizon {
            return Ok(events);
        }

        // Decay to the candidate. Nothing has been accepted there yet, so this is the
        // predictable intensity at `candidate`.
        let gap_decay = (-beta * waiting).exp();
        let mut total = 0.0;
        for m in 0..d {
            excitation[m] *= gap_decay;
        }
        for i in 0..d {
            let mut value = parameters.baseline[i];
            for j in 0..d {
                value += (parameters.excitation[i * d + j] * beta) * excitation[j];
            }
            intensity[i] = value;
            total += value;
        }

        current = candidate;
        if rng.random::<f64>() * bound <= total {
            // Superposition: choose the component proportionally to its intensity.
            let mut target = rng.random::<f64>() * total;
            let mut chosen = d - 1;
            for i in 0..d {
                target -= intensity[i];
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            events[chosen].push(candidate);
            excitation[chosen] += 1.0;
        }
    }
}

/// Outcome of a multivariate maximum-likelihood fit.
#[derive(Debug, Clone)]
pub struct Fit {
    pub parameters: Parameters,
    pub negative_log_likelihood: f64,
    pub iterations: u64,
    /// Whether the result is at a stationary point of the objective, **measured** at
    /// the result rather than claimed by the optimizer. See
    /// [`crate::univariate::Fit::converged`] for why those differ.
    pub converged: bool,
    /// Infinity norm of the per-event log-space gradient at the result.
    pub gradient_norm: f64,
    pub objective_evaluations: u64,
    pub gradient_evaluations: u64,
}

impl Fit {
    /// Spectral radius of the fitted excitation matrix, reported as a **diagnostic**.
    ///
    /// Stationarity is not enforced during optimization (CLAUDE.md §6). A fitted
    /// radius at or above 1 is a real finding about the data, not an error.
    pub fn branching_ratio_spectral_radius(&self) -> f64 {
        self.parameters.branching_ratio_spectral_radius()
    }

    pub fn is_stationary(&self) -> bool {
        self.parameters.is_stationary()
    }
}

/// Maximum-likelihood fit by L-BFGS in log-parameter space.
///
/// # Parametrization
///
/// Optimizes `(ln mu, ln alpha, ln beta)`, `alpha` included. The reasoning, and what
/// is given up by making exact zeros unreachable, is in
/// `docs/derivations/parameter_space.md`. `Parameters::new` still accepts
/// `alpha[i][j] = 0`; only `fit` is constrained in where it can land.
///
/// # Scaling
///
/// The objective is divided by the total event count, as in the univariate fit. The
/// unnormalized log-space gradient grows with `n`, and a line search whose first trial
/// step is 1 then leaps far enough that `exp` overflows — the failure recorded in
/// `docs/positioning-probe.md` part 3.
pub fn fit(observation: &Observation) -> Result<Fit, Error> {
    let d = observation.dimension();
    let total_events = observation.len();
    // `d + d*d + 1` parameters. The bound is deliberately weak: it rejects the cases
    // with no interior optimum and leaves "few events give a poor fit" to the caller.
    if total_events < d + d * d + 1 {
        return Err(Error::insufficient_data(total_events));
    }

    // Starting point. Each component's observed rate is split evenly between its own
    // baseline and the excitation it receives, so the initial guess already reproduces
    // the per-component first moment: with a uniform `alpha` of `0.5/d` per entry the
    // row sums are 0.5, and `mu/(1 - 0.5)` recovers the rate.
    let horizon = observation.horizon();
    let mut start = Vec::with_capacity(d + d * d + 1);
    for component in observation.events() {
        let rate = (component.len() as f64 / horizon).max(f64::MIN_POSITIVE);
        start.push((0.5 * rate).ln());
    }
    for _ in 0..d * d {
        start.push((0.5 / d as f64).ln());
    }
    let mean_interarrival = horizon / total_events as f64;
    start.push((1.0 / mean_interarrival).ln());

    fit_from(observation, start)
}

/// Fits from a caller-supplied starting point in **log** coordinates.
///
/// Exposed so multi-start invariance can be tested: a fit that lands somewhere
/// different depending on where it began is reporting a local optimum, and the round
/// trip would be measuring the starting point rather than the data.
pub fn fit_from(observation: &Observation, start: Vec<f64>) -> Result<Fit, Error> {
    use argmin::core::{CostFunction, Executor, Gradient as ArgminGradient, State};
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::LBFGS;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let d = observation.dimension();
    let total_events = observation.len();
    if total_events < d + d * d + 1 {
        return Err(Error::insufficient_data(total_events));
    }
    if start.len() != d + d * d + 1 {
        return Err(Error::DimensionMismatch {
            what: "start",
            actual: start.len(),
            expected: d + d * d + 1,
            dimension: d,
        });
    }

    struct Problem<'a, 'b> {
        observation: &'a Observation<'b>,
        dimension: usize,
        scale: f64,
        objective_calls: Arc<AtomicU64>,
        gradient_calls: Arc<AtomicU64>,
    }

    impl Problem<'_, '_> {
        fn parameters(&self, log_parameters: &[f64]) -> Parameters {
            let d = self.dimension;
            let baseline: Vec<f64> = log_parameters[..d].iter().map(|v| v.exp()).collect();
            let excitation: Vec<f64> = log_parameters[d..d + d * d]
                .iter()
                .map(|v| v.exp())
                .collect();
            let decay = log_parameters[d + d * d].exp();
            Parameters::new(baseline, excitation, decay).unwrap_or_else(|_| Parameters {
                baseline: vec![f64::MIN_POSITIVE; d],
                excitation: vec![f64::MIN_POSITIVE; d * d],
                decay: f64::MIN_POSITIVE,
            })
        }
    }

    impl CostFunction for Problem<'_, '_> {
        type Param = Vec<f64>;
        type Output = f64;
        fn cost(&self, log_parameters: &Self::Param) -> Result<f64, argmin::core::Error> {
            self.objective_calls.fetch_add(1, Ordering::Relaxed);
            Ok(
                negative_log_likelihood(&self.parameters(log_parameters), self.observation)
                    / self.scale,
            )
        }
    }

    impl ArgminGradient for Problem<'_, '_> {
        type Param = Vec<f64>;
        type Gradient = Vec<f64>;
        fn gradient(&self, log_parameters: &Self::Param) -> Result<Vec<f64>, argmin::core::Error> {
            self.gradient_calls.fetch_add(1, Ordering::Relaxed);
            let parameters = self.parameters(log_parameters);
            let (_, gradient) = negative_log_likelihood_and_gradient(&parameters, self.observation);
            let log_gradient = gradient.to_log_parameter_space(&parameters);
            let mut flat = log_gradient.baseline;
            flat.extend_from_slice(&log_gradient.excitation);
            flat.push(log_gradient.decay);
            for slot in &mut flat {
                *slot /= self.scale;
            }
            Ok(flat)
        }
    }

    let objective_calls = Arc::new(AtomicU64::new(0));
    let gradient_calls = Arc::new(AtomicU64::new(0));
    let problem = Problem {
        observation,
        dimension: d,
        scale: total_events as f64,
        objective_calls: Arc::clone(&objective_calls),
        gradient_calls: Arc::clone(&gradient_calls),
    };

    let line_search = MoreThuenteLineSearch::new()
        .with_c(1e-4, 0.9)
        .map_err(|e| Error::OptimizerFailed {
            message: e.to_string(),
        })?;
    let solver = LBFGS::new(line_search, 10)
        .with_tolerance_grad(1e-10)
        .map_err(|e| Error::OptimizerFailed {
            message: e.to_string(),
        })?;

    let result = Executor::new(problem, solver)
        .configure(|state| state.param(start).max_iters(1000))
        .run()
        .map_err(|e| Error::OptimizerFailed {
            message: e.to_string(),
        })?;

    let state = result.state();
    let best = state
        .get_best_param()
        .ok_or_else(|| Error::OptimizerFailed {
            message: "optimizer returned no parameters".to_owned(),
        })?;
    let baseline: Vec<f64> = best[..d].iter().map(|v| v.exp()).collect();
    let excitation: Vec<f64> = best[d..d + d * d].iter().map(|v| v.exp()).collect();
    let parameters = Parameters::new(baseline, excitation, best[d + d * d].exp())?;

    let value = negative_log_likelihood(&parameters, observation);
    let (_, gradient) = negative_log_likelihood_and_gradient(&parameters, observation);
    let log_gradient = gradient.to_log_parameter_space(&parameters);
    let events = total_events as f64;
    let gradient_norm = log_gradient
        .baseline
        .iter()
        .chain(&log_gradient.excitation)
        .chain(std::iter::once(&log_gradient.decay))
        .map(|g| (g / events).abs())
        .fold(0.0f64, f64::max);

    Ok(Fit {
        parameters,
        negative_log_likelihood: value,
        iterations: state.get_iter(),
        converged: gradient_norm < 1e-6,
        gradient_norm,
        objective_evaluations: objective_calls.load(Ordering::Relaxed),
        gradient_evaluations: gradient_calls.load(Ordering::Relaxed),
    })
}

/// Negative log-likelihood with the per-component work spread across threads.
///
/// **Bitwise identical to [`negative_log_likelihood`].** That is not a happy accident:
/// the log term is accumulated into one slot per component and combined in index
/// order, so the arithmetic is fixed regardless of how the components are scheduled.
/// `multivariate_parallel.rs` asserts it on every shape the sequential tests cover.
///
/// # What is and is not parallel
///
/// The recursion over time is inherently sequential — `B^m_r` depends on `B^m_{r-1}`
/// — so only the work *within* one distinct time is spread: advancing the `d` states
/// across the gap, and evaluating the `d` intensities. Both are independent per
/// component and neither is a reduction.
///
/// # It is much slower. Measured, not assumed.
///
/// At one distinct time the parallel work is `O(d + d^2)` floating-point operations —
/// 110 at `d = 10` — and a thread-pool dispatch costs far more than that. Median of 5,
/// Apple M2, `benchmarks/results/multivariate-parallel.json`:
///
/// | `d` | events | sequential | parallel | ratio |
/// | --- | --- | --- | --- | --- |
/// | 2 | 65 226 | 0.002844 s | 0.824304 s | 290x slower |
/// | 5 | 73 521 | 0.003419 s | 1.358337 s | 397x slower |
/// | 10 | 89 286 | 0.005522 s | 1.825924 s | 331x slower |
/// | 20 | 119 566 | 0.013534 s | 3.089594 s | 228x slower |
///
/// So this exists, is correct, is bitwise identical to the sequential path, and
/// should not be switched on. It is kept because "component-level parallelism does not
/// pay here, by two to three orders of magnitude" is a more useful thing for the
/// repository to know than an absence, and because the bit-identity machinery it
/// forced — component-major accumulation — is what any future parallel path will need.
///
/// Parallelising across time is not available: the state recursion forbids it.
/// Parallelising across independent *realizations* would work and is the shape that
/// would actually pay, but `hawk` has no multi-realization API and CLAUDE.md §5
/// forbids building one ahead of a caller.
#[cfg(feature = "rayon")]
pub fn negative_log_likelihood_parallel(parameters: &Parameters, observation: &Observation) -> f64 {
    assert_dimensions_agree(parameters, observation);
    use rayon::prelude::*;

    let d = parameters.dimension();
    let beta = parameters.decay;
    let horizon = observation.horizon();

    let mut total: f64 = 0.0;
    for i in 0..d {
        total += parameters.baseline[i] * horizon;
    }
    if observation.is_empty() {
        return total;
    }

    let mut walk = PooledWalk::new(observation.events());
    let mut counts = vec![0usize; d];
    let mut state = vec![0.0f64; d];
    let mut compensator = vec![0.0f64; d];
    let mut log_term_parts = vec![0.0f64; d];
    let mut previous_time = 0.0f64;
    let mut previous_counts = vec![0usize; d];
    let mut started = false;

    while let Some(time) = walk.next(&mut counts) {
        if started {
            let gap = time - previous_time;
            state
                .par_iter_mut()
                .zip(previous_counts.par_iter())
                .for_each(|(slot, &count)| {
                    let (advanced, _) =
                        crate::univariate::advance_excitation_state(*slot, count as f64, gap, beta);
                    *slot = advanced;
                });
        }
        started = true;

        let contribution = crate::univariate::compensator_contribution(beta, horizon - time);
        for j in 0..d {
            for _ in 0..counts[j] {
                compensator[j] += contribution;
            }
        }

        let baseline = &parameters.baseline;
        let excitation = &parameters.excitation;
        let state_view = &state;
        let counts_view = &counts;
        log_term_parts
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, part)| {
                let count = counts_view[i];
                if count == 0 {
                    return;
                }
                let mut intensity = baseline[i];
                for j in 0..d {
                    intensity += (excitation[i * d + j] * beta) * state_view[j];
                }
                let logarithm = intensity.ln();
                for _ in 0..count {
                    *part += logarithm;
                }
            });

        previous_time = time;
        previous_counts.copy_from_slice(&counts);
    }

    for i in 0..d {
        for j in 0..d {
            total += parameters.excitation[i * d + j] * compensator[j];
        }
    }
    let mut log_term = 0.0f64;
    for part in &log_term_parts {
        log_term += part;
    }
    total - log_term
}
