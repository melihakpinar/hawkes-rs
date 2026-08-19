//! Univariate exponential-kernel Hawkes process.
//!
//! Transcribed from `docs/derivations/univariate_loglikelihood.md` and
//! `docs/derivations/univariate_gradient.md`. Equation references of the form
//! `(4.4)` and `(G.6)` point there.

use crate::Error;

/// Parameters of a univariate exponential-kernel Hawkes process.
///
/// The intensity is
///
/// ```text
/// lambda(t) = mu + sum_{t_i < t} alpha * beta * exp(-beta * (t - t_i))
/// ```
///
/// with the sum strictly over earlier events, so `lambda` is predictable and
/// `lambda(t_1) = mu` (`conventions.md` C3, [Laub2015, eq. 4]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    /// `mu` — the baseline intensity.
    baseline: f64,
    /// `alpha` — the kernel's integral, and therefore the branching ratio itself
    /// (`conventions.md` C1, C2). **Not** `alpha / beta`.
    excitation: f64,
    /// `beta` — the exponential decay rate.
    decay: f64,
}

impl Parameters {
    /// Validates and constructs. All three must be strictly positive and finite
    /// (`univariate_loglikelihood.md` §1).
    ///
    /// Stationarity is deliberately **not** checked here. A non-stationary parameter
    /// set is a legitimate thing to evaluate the likelihood at, and a non-stationary
    /// *fit* is a real finding about the data rather than an error (CLAUDE.md §6).
    /// Use [`Parameters::is_stationary`] to ask.
    pub fn new(baseline: f64, excitation: f64, decay: f64) -> Result<Self, Error> {
        for (name, value) in [
            ("baseline", baseline),
            ("excitation", excitation),
            ("decay", decay),
        ] {
            if !value.is_finite() {
                return Err(Error::NonFiniteParameter { name, value });
            }
            if value <= 0.0 {
                return Err(Error::NonPositiveParameter { name, value });
            }
        }
        Ok(Self {
            baseline,
            excitation,
            decay,
        })
    }

    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    pub fn excitation(&self) -> f64 {
        self.excitation
    }

    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// The branching ratio: the mean number of direct offspring per event.
    ///
    /// Under this crate's kernel normalization the kernel integrates to `alpha`
    /// ([Laub2015, eq. 5] with `alpha_Laub = alpha * beta`), so the branching ratio
    /// *is* `alpha`. Under the other convention in the literature it would be
    /// `alpha / beta`; getting this wrong misstates stationarity by a factor of
    /// `beta`.
    pub fn branching_ratio(&self) -> f64 {
        self.excitation
    }

    /// Whether the process is stationary, i.e. branching ratio `< 1`.
    ///
    /// Reported as a diagnostic, never enforced (CLAUDE.md §6).
    pub fn is_stationary(&self) -> bool {
        self.branching_ratio() < 1.0
    }

    /// Stationary mean intensity `mu / (1 - alpha)`, or `None` when not stationary.
    ///
    /// [Laub2015, eq. 6] gives `E[lambda*(t)] -> lambda / (1 - n)` as `t -> infinity`
    /// in the defective case `n < 1`. This is CLAUDE.md §3's oracle 1.
    pub fn stationary_mean_intensity(&self) -> Option<f64> {
        self.is_stationary()
            .then(|| self.baseline / (1.0 - self.branching_ratio()))
    }
}

/// A realization observed on `[0, horizon]`, validated against the input contract.
///
/// The contract is `docs/derivations/conventions.md` C8: ascending timestamps, all
/// within `[0, horizon]` inclusive, ties permitted.
#[derive(Debug, Clone, Copy)]
pub struct Observation<'a> {
    times: &'a [f64],
    horizon: f64,
}

impl<'a> Observation<'a> {
    /// Validates and constructs.
    ///
    /// `horizon` is `T`, supplied by the caller. It is never inferred from the data:
    /// taking `T = max(times)` discards the trailing dead time and biases the
    /// baseline upward (`conventions.md` C5).
    ///
    /// Exact ties are accepted. They are evaluated under the predictable convention,
    /// where simultaneous events do not excite each other (C3, C8). Note that on tied
    /// data the objective is not a likelihood — see
    /// `univariate_loglikelihood.md` §3.1.
    pub fn new(times: &'a [f64], horizon: f64) -> Result<Self, Error> {
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(Error::InvalidHorizon { horizon });
        }
        for (index, &time) in times.iter().enumerate() {
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
            if index > 0 && time < times[index - 1] {
                return Err(Error::UnsortedEvents {
                    index,
                    previous_index: index - 1,
                    previous: times[index - 1],
                    current: time,
                });
            }
        }
        Ok(Self { times, horizon })
    }

    pub fn times(&self) -> &'a [f64] {
        self.times
    }

    pub fn horizon(&self) -> f64 {
        self.horizon
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }
}

/// Partial derivatives of the negative log-likelihood.
///
/// From `docs/derivations/univariate_gradient.md`, equations (G.1), (G.2) and (G.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gradient {
    /// `d nll / d mu` — (G.1).
    pub baseline: f64,
    /// `d nll / d alpha` — (G.2).
    pub excitation: f64,
    /// `d nll / d beta` — (G.7).
    pub decay: f64,
}

impl Gradient {
    /// Converts to log-parameter space by the chain rule `d/d ln x = x * d/dx`,
    /// equation (G.8).
    ///
    /// Positivity is enforced by optimizing over `(ln mu, ln alpha, ln beta)` rather
    /// than by constrained optimization (CLAUDE.md §6), and this is the boundary
    /// where the conversion happens. It is also where a factor silently goes missing,
    /// which is why the finite-difference check runs in both parametrizations.
    pub fn to_log_parameter_space(&self, parameters: &Parameters) -> Self {
        Self {
            baseline: parameters.baseline * self.baseline,
            excitation: parameters.excitation * self.excitation,
            decay: parameters.decay * self.decay,
        }
    }
}

/// Negative log-likelihood, `O(n)`.
///
/// Transcription of `univariate_loglikelihood.md` (4.5). See
/// [`negative_log_likelihood_and_gradient`] for the shared implementation and for
/// what the recursion does about tied timestamps.
pub fn negative_log_likelihood(parameters: &Parameters, observation: &Observation) -> f64 {
    negative_log_likelihood_and_gradient(parameters, observation).0
}

/// Negative log-likelihood and its analytic gradient, in one `O(n)` pass.
///
/// Transcription of `univariate_loglikelihood.md` §5 and `univariate_gradient.md` §5.
/// The two are computed together because both need the excitation state `B_j`, and
/// keeping them in one pass is also what stops them from drifting apart.
///
/// # The recursion groups by distinct time
///
/// The textbook form [Laub2015, eq. 20], `A_k = exp(-beta*(t_k - t_{k-1}))*(1 + A_{k-1})`,
/// is **wrong** on tied input: at a tie it evaluates `exp(0) = 1` and counts the
/// simultaneous event as exciting, which `conventions.md` C3 forbids. [Laub2015]
/// derives it for a *simple* point process, where ties cannot occur, so this is a
/// hypothesis that does not survive `hawk`'s input contract (C8) rather than an error
/// in the paper. On `t = [1, 2, 2, 3]` the difference is about 9%.
///
/// The grouped form (4.4) is used instead:
///
/// ```text
/// B_j = exp(-beta*d_j) * ( B_{j-1} + c_{j-1} )
/// ```
///
/// where `s_j` are the *distinct* times and `c_j` their multiplicities. It reduces to
/// the textbook form when every `c_j` is 1. `count_at_previous_time` carries `c_{j-1}`
/// so no grouping pass is needed.
///
/// # On tied data this is not a likelihood
///
/// It remains the correct value of the expression, and it still agrees with `tick`.
/// But [Laub2015, Theorem 3] is the likelihood of a simple point process, so on tied
/// input the maximum-likelihood asymptotics do not apply. See
/// `univariate_loglikelihood.md` §3.1.
pub fn negative_log_likelihood_and_gradient(
    parameters: &Parameters,
    observation: &Observation,
) -> (f64, Gradient) {
    let mu = parameters.baseline;
    let alpha = parameters.excitation;
    let beta = parameters.decay;
    let horizon = observation.horizon();
    let times = observation.times();

    if times.is_empty() {
        // No events: no log term, nothing to excite. nll = mu*T, and the excitation
        // parameters cannot influence it.
        return (
            mu * horizon,
            Gradient {
                baseline: horizon,
                excitation: 0.0,
                decay: 0.0,
            },
        );
    }

    // sum_j c_j * ( 1 - exp(-beta*(T - s_j)) )
    let mut compensator_excitation = 0.0;
    // sum_j c_j * log(lambda_j)
    let mut log_term = 0.0;
    // sum_j c_j / lambda_j
    let mut baseline_accumulator = 0.0;
    // sum_j c_j * beta * B_j / lambda_j
    let mut excitation_accumulator = 0.0;
    // sum_j c_j * W_j * exp(-beta*W_j)
    let mut decay_compensator_accumulator = 0.0;
    // sum_j c_j * ( B_j + beta*Bp_j ) / lambda_j
    let mut decay_log_accumulator = 0.0;

    // B_j, the excitation state at the current distinct time. Base case (4.3).
    let mut excitation_state = 0.0;
    // Bp_j = d B_j / d beta. Base case (G.5).
    let mut excitation_state_derivative = 0.0;

    let mut previous_time = times[0];
    let mut count_at_previous_time = 0.0f64;

    for &time in times {
        if time != previous_time {
            // A new distinct time s_j. Advance the state across the gap.
            let gap = time - previous_time;
            let gap_decay = (-beta * gap).exp();
            let advanced = gap_decay * (excitation_state + count_at_previous_time); // (4.4)
            // (G.6). Must use the advanced value, not the previous one, and must be
            // evaluated before `excitation_state` is overwritten.
            excitation_state_derivative = -gap * advanced + gap_decay * excitation_state_derivative;
            excitation_state = advanced;
            previous_time = time;
            count_at_previous_time = 0.0;
        }

        let intensity = mu + alpha * beta * excitation_state;
        let window = horizon - time;
        let window_decay = (-beta * window).exp();

        log_term += intensity.ln();
        // `-exp_m1(-x)` rather than `1 - exp(-x)`: for events near the horizon the
        // direct form loses precision to cancellation (§5's numerical notes).
        compensator_excitation += -(-beta * window).exp_m1();
        baseline_accumulator += 1.0 / intensity;
        excitation_accumulator += beta * excitation_state / intensity;
        decay_compensator_accumulator += window * window_decay;
        decay_log_accumulator +=
            (excitation_state + beta * excitation_state_derivative) / intensity;

        count_at_previous_time += 1.0;
    }

    let negative_log_likelihood = mu * horizon + alpha * compensator_excitation - log_term;
    let gradient = Gradient {
        baseline: horizon - baseline_accumulator, // (G.1)
        excitation: compensator_excitation - excitation_accumulator, // (G.2)
        decay: alpha * (decay_compensator_accumulator - decay_log_accumulator), // (G.7)
    };
    (negative_log_likelihood, gradient)
}
