//! `hawk` side of positioning probe part 2: inner-loop cost, and how many objective
//! evaluations a fit performs. See `docs/positioning-probe.md` part 2.

use std::fs;
use std::hint::black_box;
use std::time::Instant;

use hawk::univariate::{
    Observation, Parameters, fit, negative_log_likelihood_and_gradient, simulate,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_819;
const BASELINE: f64 = 0.5;
const EXCITATION: f64 = 0.6;
const DECAY: f64 = 1.0;
const WARMUP: usize = 1;
const TIMED: usize = 5;

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .expect("usage: inner_loop_probe <dir>");
    fs::create_dir_all(&out_dir).unwrap();

    let truth = Parameters::new(BASELINE, EXCITATION, DECAY).unwrap();
    let rate = truth.stationary_mean_intensity().unwrap();
    let mut records = Vec::new();

    for nominal in [10_000usize, 100_000, 1_000_000] {
        let horizon = nominal as f64 / rate;
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let times = simulate(&truth, horizon, &mut rng).unwrap();

        let path = format!("{out_dir}/events_{nominal}.txt");
        let mut body = String::with_capacity(times.len() * 20);
        body.push_str(&format!("{horizon:?}\n"));
        for t in &times {
            body.push_str(&format!("{t:?}\n"));
        }
        fs::write(&path, body).unwrap();

        let observation = Observation::new(&times, horizon).unwrap();

        // Single evaluation: value and analytic gradient, one pass. Evaluated at the
        // true parameters so both sides evaluate at the same point.
        for _ in 0..WARMUP {
            black_box(negative_log_likelihood_and_gradient(&truth, &observation));
        }
        let mut elapsed = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let start = Instant::now();
            let value = negative_log_likelihood_and_gradient(&truth, &observation);
            elapsed.push(start.elapsed().as_secs_f64());
            black_box(value);
        }
        elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Evaluation count during a full fit. Deterministic, so one run suffices.
        let fitted = fit(&observation).unwrap();

        eprintln!(
            "n={:<9} events={:<9} eval_median={:.9}s  fit: iters={} obj_evals={} \
             grad_evals={}",
            nominal,
            times.len(),
            elapsed[TIMED / 2],
            fitted.iterations,
            fitted.objective_evaluations,
            fitted.gradient_evaluations,
        );

        records.push(format!(
            r#"    {{
      "nominal_n": {nominal},
      "events": {},
      "eval_seconds_median": {:?},
      "eval_seconds_min": {:?},
      "eval_seconds_max": {:?},
      "eval_seconds_all": [{}],
      "fit_iterations": {},
      "fit_objective_evaluations": {},
      "fit_gradient_evaluations": {}
    }}"#,
            times.len(),
            elapsed[TIMED / 2],
            elapsed[0],
            elapsed[TIMED - 1],
            elapsed
                .iter()
                .map(|v| format!("{v:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            fitted.iterations,
            fitted.objective_evaluations,
            fitted.gradient_evaluations,
        ));
    }

    fs::write(
        format!("{out_dir}/hawk_inner.json"),
        format!(
            "{{\n  \"side\": \"hawk\",\n  \"warmup\": {WARMUP},\n  \"timed\": {TIMED},\n  \
             \"evaluated_at\": {{\"baseline\": {BASELINE:?}, \"excitation\": \
             {EXCITATION:?}, \"decay\": {DECAY:?}}},\n  \"runs\": [\n{}\n  ]\n}}\n",
            records.join(",\n")
        ),
    )
    .unwrap();
}
