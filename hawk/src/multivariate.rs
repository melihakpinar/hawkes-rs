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
    /// Power iteration tightens the bracket. It is run on `A + I` rather than `A`,
    /// and 1 subtracted at the end: `rho(A + I) = rho(A) + 1` for non-negative `A`,
    /// and shifting makes the matrix aperiodic. Without the shift a periodic matrix
    /// such as `[[0, 2], [0.9, 0]]` oscillates forever between the bounds `0.9` and
    /// `2` and never converges — that case is in the tests.
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
            // Both bounds are rigorous at every step, so stopping early is safe.
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
        // Midpoint of the final bracket, shifted back.
        (0.5 * (lower + upper) - 1.0).max(0.0)
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
    let mut log_term = 0.0f64;
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
                log_term += logarithm;
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
    let mut log_term = 0.0f64;
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
                log_term += logarithm;
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
