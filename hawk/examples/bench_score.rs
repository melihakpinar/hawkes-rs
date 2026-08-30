//! Scores `tick`'s returned parameters under `hawk`'s unpenalized objective, so two
//! answers from two different objectives sit in one common unit.
//! See `benchmarks/README.md` §5.5.
//!
//! Usage: `bench_score <dir> <d>`

use std::fs;

use hawk::multivariate::{Observation, Parameters, negative_log_likelihood};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct Run {
    dimension: usize,
    nominal_n: usize,
    completed: bool,
    #[serde(default)]
    baseline: Vec<f64>,
    #[serde(default)]
    excitation: Vec<f64>,
    #[serde(default)]
    decay: f64,
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize, Serialize)]
struct Side {
    runs: Vec<Run>,
    #[serde(flatten)]
    rest: serde_json::Map<String, serde_json::Value>,
}

fn read_events(path: &str) -> (f64, Vec<Vec<f64>>) {
    let raw = fs::read_to_string(path).expect("events file");
    let mut values = raw.split_whitespace();
    let horizon: f64 = values.next().unwrap().parse().unwrap();
    let d: usize = values.next().unwrap().parse().unwrap();
    let mut events = Vec::with_capacity(d);
    for _ in 0..d {
        let count: usize = values.next().unwrap().parse().unwrap();
        events.push(
            (0..count)
                .map(|_| values.next().unwrap().parse().unwrap())
                .collect(),
        );
    }
    (horizon, events)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: bench_score <dir> <d> <n>");
    let d: usize = args.next().expect("dimension").parse().unwrap();
    let n: usize = args.next().expect("event count").parse().unwrap();

    let path = format!("{dir}/tick_d{d}_n{n}.json");
    let mut side: Side = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();

    for run in &mut side.runs {
        if !run.completed {
            continue;
        }
        let (horizon, events) = read_events(&format!(
            "{dir}/events_d{}_n{}.txt",
            run.dimension, run.nominal_n
        ));
        let observation = Observation::new(&events, horizon).unwrap();
        let parameters =
            Parameters::new(run.baseline.clone(), run.excitation.clone(), run.decay).unwrap();
        let value = negative_log_likelihood(&parameters, &observation).unwrap();
        let spectral = parameters.branching_ratio_spectral_radius();
        run.rest.insert(
            "negative_log_likelihood_under_hawk_objective".to_owned(),
            serde_json::json!(value),
        );
        run.rest
            .insert("spectral_radius".to_owned(), serde_json::json!(spectral));
        eprintln!(
            "d={} n={} scored under hawk nll = {value:.6} (rho={spectral:.6})",
            run.dimension, run.nominal_n
        );
    }

    fs::write(&path, serde_json::to_string_pretty(&side).unwrap() + "\n").unwrap();
}
