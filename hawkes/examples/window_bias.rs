//! Measures what OQ-5 resolved by reading source: `tick`'s learner takes no
//! `end_times`, so its baseline cannot depend on the observation window.
//!
//! One realization, fitted repeatedly under a growing *declared* horizon. The events
//! never change; only the length of the window the caller says was observed. Dead time
//! is the interval after the last event during which the process was watched and
//! produced nothing.
//!
//! `mu` is a rate, so the correct estimate must fall as the declared window grows.
//! An estimator that cannot see the window returns the same number every time, and is
//! therefore wrong by a factor that grows with the dead time.
//!
//! Usage: `window_bias <out_dir>`

use std::fs;
use std::io::Write;

use hawkes::univariate::{Observation, Parameters, fit, simulate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_831;
const BASELINE: f64 = 0.5;
const EXCITATION: f64 = 0.6;
const DECAY: f64 = 1.0;
const NOMINAL_EVENTS: usize = 20_000;
const DEAD_TIME_FRACTIONS: [f64; 5] = [0.0, 0.05, 0.10, 0.25, 0.50];

fn main() {
    let out_dir = std::env::args().nth(1).expect("usage: window_bias <dir>");
    fs::create_dir_all(&out_dir).unwrap();

    let truth = Parameters::new(BASELINE, EXCITATION, DECAY).unwrap();
    let observed_horizon = NOMINAL_EVENTS as f64 / truth.stationary_mean_intensity().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let times = simulate(&truth, observed_horizon, &mut rng).unwrap();
    let last_event = *times.last().unwrap();

    let path = format!("{out_dir}/window_bias_events.txt");
    let mut file = fs::File::create(&path).unwrap();
    writeln!(file, "{observed_horizon:?}").unwrap();
    writeln!(file, "1").unwrap();
    writeln!(file, "{}", times.len()).unwrap();
    for t in &times {
        writeln!(file, "{t:?}").unwrap();
    }
    drop(file);

    let mut rows = Vec::new();
    for fraction in DEAD_TIME_FRACTIONS {
        let declared = observed_horizon * (1.0 + fraction);
        let observation = Observation::new(&times, declared).unwrap();
        let result = fit(&observation).unwrap();
        eprintln!(
            "dead_time={:5.0}% declared_T={declared:10.2} hawkes baseline={:.6}",
            fraction * 100.0,
            result.parameters.baseline()
        );
        rows.push(format!(
            r#"{{"dead_time_fraction":{fraction:?},"declared_horizon":{declared:?},"baseline":{:?},"excitation":{:?},"decay":{:?},"converged":{}}}"#,
            result.parameters.baseline(),
            result.parameters.excitation(),
            result.parameters.decay(),
            result.converged
        ));
    }

    let body = format!(
        r#"{{"library":"hawkes","experiment":"window_bias","question":"OQ-5","seed":{SEED},"events":{},"observed_horizon":{observed_horizon:?},"last_event":{last_event:?},"true_baseline":{BASELINE:?},"true_excitation":{EXCITATION:?},"true_decay":{DECAY:?},"rows":[{}]}}"#,
        times.len(),
        rows.join(",")
    );
    fs::write(format!("{out_dir}/hawk_window_bias.json"), body + "\n").unwrap();
}
