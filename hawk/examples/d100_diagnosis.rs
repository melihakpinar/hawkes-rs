//! Two separate questions about the `d = 100` benchmark result, measured rather than
//! guessed.
//!
//! **A. How does one evaluation scale with `d`?** The gradient pass is `O(K * d^2)` for
//! `K` distinct pooled times, so with the event count held fixed the time should grow
//! like `d^2`. Anything materially worse is a defect in the implementation, not a
//! property of the problem.
//!
//! **B. Why does the fit not converge?** `fit` at `d = 100` stopped at its 1000
//! iteration cap with a gradient norm of 3.5e-6 against a 1e-6 threshold. A flat
//! plateau and a slow steady descent are different diagnoses: the first says the
//! optimizer is stuck, the second says the cap is too low. This records the trace.
//!
//! The optimizer here is assembled from the public API to be identical to
//! `multivariate::fit_from`: same starting point, same per-event scaling, same
//! More-Thuente line search, same 10 correction pairs, same `1e-10` gradient tolerance,
//! same 1000 iteration cap. If that assembly ever drifts from `fit`, part B stops
//! describing `fit`.
//!
//! Usage: `d100_diagnosis <out_dir>`

use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use argmin::core::observers::{Observe, ObserverMode};
use argmin::core::{CostFunction, Executor, Gradient as ArgminGradient, IterState, KV, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use hawk::multivariate::{Observation, Parameters, negative_log_likelihood_and_gradient, simulate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_831;
const DECAY: f64 = 1.0;
/// Held fixed across `d` so the only thing varying is the dimension.
const EVENTS: usize = 100_000;
const EVAL_REPETITIONS: usize = 5;

fn truth(d: usize) -> Parameters {
    Parameters::new(vec![0.5 / d as f64; d], vec![0.6 / d as f64; d * d], DECAY)
        .expect("valid parameters")
}

fn data(d: usize) -> (f64, Vec<Vec<f64>>) {
    let truth = truth(d);
    let rate: f64 = truth
        .stationary_mean_intensity()
        .expect("stationary")
        .iter()
        .sum();
    let horizon = EVENTS as f64 / rate;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    (horizon, simulate(&truth, horizon, &mut rng).unwrap())
}

// --- Part B: the optimizer, replicated from `multivariate::fit_from` ----------------

struct Problem<'a, 'b> {
    observation: &'a Observation<'b>,
    dimension: usize,
    scale: f64,
}

impl Problem<'_, '_> {
    fn parameters(&self, log_parameters: &[f64]) -> Parameters {
        let d = self.dimension;
        let baseline: Vec<f64> = log_parameters[..d].iter().map(|v| v.exp()).collect();
        let excitation: Vec<f64> = log_parameters[d..d + d * d]
            .iter()
            .map(|v| v.exp())
            .collect();
        Parameters::new(baseline, excitation, log_parameters[d + d * d].exp())
            .expect("positive by construction")
    }
}

impl CostFunction for Problem<'_, '_> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, log_parameters: &Self::Param) -> Result<f64, argmin::core::Error> {
        let (value, _) = negative_log_likelihood_and_gradient(
            &self.parameters(log_parameters),
            self.observation,
        )?;
        Ok(value / self.scale)
    }
}

impl ArgminGradient for Problem<'_, '_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, log_parameters: &Self::Param) -> Result<Vec<f64>, argmin::core::Error> {
        let parameters = self.parameters(log_parameters);
        let (_, gradient) = negative_log_likelihood_and_gradient(&parameters, self.observation)?;
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

/// Records the infinity norm of the scaled log-space gradient at each iteration —
/// exactly the quantity `Fit::gradient_norm` reports and `converged` thresholds at 1e-6.
#[derive(Clone)]
struct Trace(Arc<Mutex<Vec<(u64, f64, f64)>>>);

/// Bound to the concrete state L-BFGS uses here: the generic `State` trait carries no
/// gradient, only `IterState` does.
type LbfgsState = IterState<Vec<f64>, Vec<f64>, (), (), (), f64>;

impl Observe<LbfgsState> for Trace {
    fn observe_iter(&mut self, state: &LbfgsState, _kv: &KV) -> Result<(), argmin::core::Error> {
        let norm = state
            .get_gradient()
            .map(|g| g.iter().fold(0.0f64, |acc, v| acc.max(v.abs())))
            .unwrap_or(f64::NAN);
        self.0
            .lock()
            .unwrap()
            .push((state.get_iter(), state.get_cost(), norm));
        Ok(())
    }
}

fn starting_point(observation: &Observation) -> Vec<f64> {
    let d = observation.dimension();
    let horizon = observation.horizon();
    let mut start = Vec::with_capacity(d + d * d + 1);
    for component in observation.events() {
        let rate = (component.len() as f64 / horizon).max(f64::MIN_POSITIVE);
        start.push((0.5 * rate).ln());
    }
    for _ in 0..d * d {
        start.push((0.5 / d as f64).ln());
    }
    let mean_interarrival = horizon / observation.len() as f64;
    start.push((1.0 / mean_interarrival).ln());
    start
}

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: d100_diagnosis <dir>");
    fs::create_dir_all(&out_dir).unwrap();

    // --- Part A -------------------------------------------------------------------
    let mut scaling = Vec::new();
    for d in [1usize, 3, 10, 30, 100] {
        let (horizon, events) = data(d);
        let observation = Observation::new(&events, horizon).unwrap();
        let parameters = truth(d);
        let realized: usize = events.iter().map(|c| c.len()).sum();

        std::hint::black_box(
            negative_log_likelihood_and_gradient(&parameters, &observation).unwrap(),
        );
        let mut times = Vec::with_capacity(EVAL_REPETITIONS);
        for _ in 0..EVAL_REPETITIONS {
            let start = Instant::now();
            std::hint::black_box(
                negative_log_likelihood_and_gradient(&parameters, &observation).unwrap(),
            );
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[EVAL_REPETITIONS / 2];
        eprintln!(
            "d={d:<4} events={realized:<7} one nll+gradient = {median:.6}s  \
             per event per d^2 = {:.3e}",
            median / realized as f64 / (d * d) as f64
        );
        scaling.push(format!(
            r#"{{"dimension":{d},"events":{realized},"seconds_median":{median:?},"seconds_min":{:?},"seconds_max":{:?}}}"#,
            times[0],
            times[EVAL_REPETITIONS - 1]
        ));
    }

    // --- Part B -------------------------------------------------------------------
    let (horizon, events) = data(100);
    let observation = Observation::new(&events, horizon).unwrap();
    let trace = Trace(Arc::new(Mutex::new(Vec::new())));
    let problem = Problem {
        observation: &observation,
        dimension: 100,
        scale: observation.len() as f64,
    };
    let line_search = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
    let solver = LBFGS::new(line_search, 10)
        .with_tolerance_grad(1e-10)
        .unwrap();
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(starting_point(&observation)).max_iters(1000))
        .add_observer(trace.clone(), ObserverMode::Always)
        .run()
        .unwrap();
    eprintln!(
        "d=100 finished: {}",
        result.state().get_termination_status()
    );

    let recorded = trace.0.lock().unwrap();
    let rows: Vec<String> = recorded
        .iter()
        .map(|(iter, cost, norm)| {
            format!(r#"{{"iteration":{iter},"cost":{cost:?},"gradient_norm":{norm:?}}}"#)
        })
        .collect();
    for (iter, cost, norm) in recorded.iter() {
        if *iter <= 5 || iter % 100 == 0 || *iter == recorded.len() as u64 {
            eprintln!("  iter {iter:<5} cost {cost:.9}  |grad|_inf {norm:.6e}");
        }
    }

    let body = format!(
        r#"{{"experiment":"d100_diagnosis","seed":{SEED},"events_target":{EVENTS},"evaluation_scaling":[{}],"d100_trace":[{}]}}"#,
        scaling.join(","),
        rows.join(",")
    );
    fs::write(format!("{out_dir}/d100-diagnosis.json"), body + "\n").unwrap();
}
