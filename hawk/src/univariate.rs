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

/// The compensator `Lambda(t_k) = int_0^{t_k} lambda(u) du`, evaluated at every event.
///
/// ```text
/// Lambda(t) = mu*t + alpha * sum_{t_i < t} ( 1 - exp(-beta*(t - t_i)) )
/// ```
///
/// Writing `m_j` for the number of events strictly before the distinct time `s_j` and
/// `B_j` for the excitation state (4.4), the sum telescopes to `m_j - B_j`, so this
/// runs in `O(n)` on the same recursion the likelihood uses.
///
/// Its purpose is Ogata residual analysis: by the random time change theorem
/// [Laub2015, Theorem 4], if the parameters are correct then
/// `{Lambda(t_1), ..., Lambda(t_n)}` is a realization of a unit-rate Poisson process,
/// so the successive differences are i.i.d. `Exp(1)`. That is CLAUDE.md §3's oracle 2,
/// and it validates the simulator and the intensity jointly — they cannot both be
/// wrong in a way that still produces a unit-rate Poisson process.
///
/// Events at or after `t` contribute nothing: a simultaneous event gives
/// `1 - exp(0) = 0`, so ties need no special handling here.
pub fn compensator_at_events(parameters: &Parameters, observation: &Observation) -> Vec<f64> {
    let mu = parameters.baseline;
    let alpha = parameters.excitation;
    let beta = parameters.decay;
    let times = observation.times();

    let mut compensators = Vec::with_capacity(times.len());
    if times.is_empty() {
        return compensators;
    }

    let mut excitation_state = 0.0f64;
    let mut events_strictly_before = 0.0f64;
    let mut previous_time = times[0];
    let mut count_at_previous_time = 0.0f64;

    for &time in times {
        if time != previous_time {
            let gap = time - previous_time;
            excitation_state = (-beta * gap).exp() * (excitation_state + count_at_previous_time);
            events_strictly_before += count_at_previous_time;
            previous_time = time;
            count_at_previous_time = 0.0;
        }
        compensators.push(mu * time + alpha * (events_strictly_before - excitation_state));
        count_at_previous_time += 1.0;
    }
    compensators
}

/// Simulates a realization on `[0, horizon]` by Ogata's modified thinning algorithm.
///
/// [Laub2015, Algorithm 2]. The intensity is non-increasing between arrivals, so
/// `lambda(t+)` — the value just after the last accepted event — bounds it until the
/// next one, and can be used as the thinning rate `M`.
///
/// # Difference from the published algorithm
///
/// [Laub2015, Algorithm 2] obtains the bound as `lambda*(t + epsilon)` with
/// `epsilon = 1e-10`, because it treats `lambda*` as a black box. `hawk` carries the
/// excitation state explicitly, so the right-hand limit is available exactly:
/// accepting an event adds `exp(0) = 1` to the state. The `epsilon` nudge is
/// therefore unnecessary, and dropping it removes a small bias — with `epsilon > 0`
/// the bound is evaluated slightly *after* `t` and is slightly too low, which makes
/// the thinning accept marginally too often.
///
/// # Non-stationary parameters
///
/// No cap is imposed on the number of events. With a branching ratio `>= 1`
/// ([`Parameters::is_stationary`]) the process is explosive and a long horizon can
/// produce an arbitrarily large realization. That is the honest behaviour of the
/// model rather than an error, but callers passing unvalidated parameters should
/// check stationarity first.
pub fn simulate(
    parameters: &Parameters,
    horizon: f64,
    rng: &mut impl rand::Rng,
) -> Result<Vec<f64>, Error> {
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err(Error::InvalidHorizon { horizon });
    }
    let mu = parameters.baseline;
    let alpha = parameters.excitation;
    let beta = parameters.decay;

    let mut times = Vec::new();
    let mut current = 0.0f64;
    // sum_{t_i <= current} exp(-beta*(current - t_i)), the state of the RIGHT limit,
    // so `mu + alpha*beta*excitation` is `lambda(current+)`.
    let mut excitation = 0.0f64;

    loop {
        // Bound for the interval starting at `current`: the intensity decays from
        // here until the next accepted event, so this dominates it throughout.
        let bound = mu + alpha * beta * excitation;

        // Exp(bound) waiting time. `bound >= mu > 0`, so this is always finite.
        let uniform: f64 = rng.random::<f64>();
        let waiting = -uniform.ln() / bound;
        let candidate = current + waiting;
        if candidate >= horizon {
            return Ok(times);
        }

        // Decay the state to the candidate. No event has been accepted at
        // `candidate` yet, so this is the predictable intensity there.
        excitation *= (-beta * waiting).exp();
        let intensity = mu + alpha * beta * excitation;

        current = candidate;
        if rng.random::<f64>() * bound <= intensity {
            times.push(candidate);
            // The accepted event contributes exp(0) = 1 to the right limit.
            excitation += 1.0;
        }
    }
}

/// Outcome of a maximum-likelihood fit.
#[derive(Debug, Clone)]
pub struct Fit {
    /// The fitted parameters.
    pub parameters: Parameters,
    /// Negative log-likelihood at the optimum.
    pub negative_log_likelihood: f64,
    /// Iterations the optimizer took.
    pub iterations: u64,
    /// Whether the fit is at a stationary point of the objective.
    ///
    /// Determined by measuring the gradient at the result, not by asking the
    /// optimizer whether it thinks it finished. Those differ: a line search that
    /// fails on its first trial step makes L-BFGS return its own starting point after
    /// one iteration, which is indistinguishable from success if the only question
    /// asked is "did it stop before the iteration cap?".
    ///
    /// A `false` here is a warning about the fit, not an error: the parameters are
    /// still the best point found, and the caller can decide what to do about it.
    pub converged: bool,

    /// Infinity norm of the per-event log-space gradient at the result, the quantity
    /// [`Fit::converged`] thresholds.
    pub gradient_norm: f64,

    /// Number of objective evaluations, i.e. calls to the negative log-likelihood.
    ///
    /// Counts **evaluations, not iterations**. A quasi-Newton iteration performs one
    /// or more line-search trials, and each trial is an evaluation, so this is
    /// strictly larger than [`Fit::iterations`] and is the quantity that determines
    /// how much work a fit actually does.
    pub objective_evaluations: u64,

    /// Number of gradient evaluations. Counted separately from
    /// [`Fit::objective_evaluations`] because the solver does not necessarily request
    /// both at every point.
    pub gradient_evaluations: u64,
}

impl Fit {
    /// Branching ratio of the fitted parameters, reported as a **diagnostic**.
    ///
    /// Stationarity is not enforced during optimization (CLAUDE.md §6). A fitted
    /// branching ratio at or above 1 is a real finding about the data — the sample
    /// looks explosive — and not an error to be suppressed. Check it.
    pub fn branching_ratio(&self) -> f64 {
        self.parameters.branching_ratio()
    }

    /// Whether the fitted process is stationary. See [`Fit::branching_ratio`].
    pub fn is_stationary(&self) -> bool {
        self.parameters.is_stationary()
    }
}

/// Maximum-likelihood fit by L-BFGS in log-parameter space.
///
/// # Parametrization
///
/// The optimizer works on `(ln mu, ln alpha, ln beta)`, so positivity holds by
/// construction and no constrained solver is needed (CLAUDE.md §6). Conversion
/// happens at the boundary via (G.8).
///
/// # Stationarity
///
/// Not enforced. The optimizer may pass through, and settle at, a branching ratio
/// above 1. See [`Fit::branching_ratio`].
///
/// # On tied data
///
/// This is still the minimizer of the objective, but it is not a maximum-likelihood
/// estimator in the sense the asymptotic theory requires, because on tied data the
/// objective is not a likelihood ([Laub2015, Theorem 3] is stated for a *simple*
/// point process). See `docs/derivations/univariate_loglikelihood.md` §3.1. The
/// numbers are still the numbers; the guarantees are not.
pub fn fit(observation: &Observation) -> Result<Fit, Error> {
    use argmin::core::{CostFunction, Executor, Gradient as ArgminGradient, State};
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::LBFGS;

    // Three parameters cannot be identified from a handful of events. The bound is
    // deliberately low: it rejects the degenerate cases where the objective has no
    // interior optimum at all, and leaves "few events give a poor fit" to the caller,
    // since that is a question about precision rather than about validity.
    if observation.len() < 3 {
        return Err(Error::insufficient_data(observation.len()));
    }

    struct Problem<'a, 'b> {
        observation: &'a Observation<'b>,
        objective_calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
        gradient_calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
        /// The objective is divided by the event count, so the optimizer minimizes
        /// the negative log-likelihood **per event**.
        ///
        /// This is a scaling choice, not a modelling one: the minimizer is identical.
        /// It matters because the gradient of the unnormalized objective grows
        /// linearly in `n` — around 800 for a 16000-event realization — and a line
        /// search whose first trial step is 1 then leaps a vast distance in log
        /// space, where `exp` promptly overflows. The search fails on its first
        /// iteration and L-BFGS returns its starting point, which looks exactly like
        /// convergence. `tick` normalizes its loss by the jump count for the same
        /// reason (`conventions.md` C7).
        scale: f64,
    }

    impl Problem<'_, '_> {
        fn parameters(&self, log_parameters: &[f64]) -> Parameters {
            // exp() of a finite f64 is >= 0; the only way to reach 0 is underflow at
            // about -745, which the line search does not visit in practice. Falling
            // back keeps the closure total rather than panicking inside the solver.
            Parameters::new(
                log_parameters[0].exp(),
                log_parameters[1].exp(),
                log_parameters[2].exp(),
            )
            .unwrap_or(Parameters {
                baseline: f64::MIN_POSITIVE,
                excitation: f64::MIN_POSITIVE,
                decay: f64::MIN_POSITIVE,
            })
        }
    }

    impl CostFunction for Problem<'_, '_> {
        type Param = Vec<f64>;
        type Output = f64;

        fn cost(&self, log_parameters: &Self::Param) -> Result<f64, argmin::core::Error> {
            self.objective_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            self.gradient_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let parameters = self.parameters(log_parameters);
            let (_, gradient) = negative_log_likelihood_and_gradient(&parameters, self.observation);
            let log_gradient = gradient.to_log_parameter_space(&parameters);
            Ok(vec![
                log_gradient.baseline / self.scale,
                log_gradient.excitation / self.scale,
                log_gradient.decay / self.scale,
            ])
        }
    }

    // Starting point. The observed event rate is n/T; splitting it evenly between
    // baseline and excitation gives `mu = rate/2` with `alpha = 0.5`, whose
    // stationary mean intensity `mu/(1-alpha)` is exactly `rate` -- so the initial
    // guess already reproduces the first moment. `beta` starts at the reciprocal of
    // the mean inter-arrival time, which is the only timescale in the data.
    let horizon = observation.horizon();
    let rate = observation.len() as f64 / horizon;
    let mean_interarrival = horizon / observation.len() as f64;
    let start = vec![
        (0.5 * rate).ln(),
        0.5f64.ln(),
        (1.0 / mean_interarrival).ln(),
    ];

    let line_search = MoreThuenteLineSearch::new()
        .with_c(1e-4, 0.9)
        .map_err(|e| Error::OptimizerFailed {
            message: e.to_string(),
        })?;
    // Seven correction pairs is argmin's usual default and is far more history than a
    // three-parameter problem can use; L-BFGS here is effectively BFGS.
    let solver = LBFGS::new(line_search, 7)
        .with_tolerance_grad(1e-10)
        .map_err(|e| Error::OptimizerFailed {
            message: e.to_string(),
        })?;

    let objective_calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let gradient_calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let problem = Problem {
        observation,
        objective_calls: std::sync::Arc::clone(&objective_calls),
        gradient_calls: std::sync::Arc::clone(&gradient_calls),
        scale: observation.len() as f64,
    };
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(start).max_iters(500))
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
    let parameters = Parameters::new(best[0].exp(), best[1].exp(), best[2].exp())?;
    let negative_log_likelihood = negative_log_likelihood(&parameters, observation);

    // Measure convergence rather than infer it. See `Fit::converged`.
    let (_, gradient) = negative_log_likelihood_and_gradient(&parameters, observation);
    let log_gradient = gradient.to_log_parameter_space(&parameters);
    let events = observation.len() as f64;
    let gradient_norm = (log_gradient.baseline / events)
        .abs()
        .max((log_gradient.excitation / events).abs())
        .max((log_gradient.decay / events).abs());

    Ok(Fit {
        parameters,
        negative_log_likelihood,
        iterations: state.get_iter(),
        // Threshold on the per-event gradient, so it does not tighten as the sample
        // grows. 1e-6 is far below the sampling noise in the parameters themselves --
        // the standard errors are orders of magnitude larger -- so a fit that meets
        // it is at the optimum for every practical purpose.
        converged: gradient_norm < 1e-6,
        gradient_norm,
        objective_evaluations: objective_calls.load(std::sync::atomic::Ordering::Relaxed),
        gradient_evaluations: gradient_calls.load(std::sync::atomic::Ordering::Relaxed),
    })
}
