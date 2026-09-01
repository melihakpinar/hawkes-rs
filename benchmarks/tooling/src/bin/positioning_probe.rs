//! `hawkes` side of the positioning probe. See `docs/positioning-probe.md`.
//!
//! Generates the shared data, writes it out for the `tick` side, and times the fit.

use std::fs;
use std::io::Write;
use std::time::Instant;

use hawkes::univariate::{Observation, Parameters, fit, simulate};
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
        .expect("usage: positioning_probe <dir>");
    fs::create_dir_all(&out_dir).unwrap();

    let truth = Parameters::new(BASELINE, EXCITATION, DECAY).unwrap();
    let rate = truth.stationary_mean_intensity().unwrap();

    let mut records = Vec::new();
    for nominal in [1_000usize, 10_000, 100_000, 1_000_000] {
        let horizon = nominal as f64 / rate;
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let times = simulate(&truth, horizon, &mut rng).unwrap();

        // Share the exact events with the tick side. `{:?}` on f64 is
        // shortest-round-trip, so Python's float() reads back the identical value.
        let path = format!("{out_dir}/events_{nominal}.txt");
        let mut file = fs::File::create(&path).unwrap();
        writeln!(file, "{horizon:?}").unwrap();
        for t in &times {
            writeln!(file, "{t:?}").unwrap();
        }
        drop(file);

        let observation = Observation::new(&times, horizon).unwrap();

        for _ in 0..WARMUP {
            std::hint::black_box(fit(&observation).unwrap());
        }
        let mut elapsed = Vec::with_capacity(TIMED);
        let mut last = None;
        for _ in 0..TIMED {
            let start = Instant::now();
            let result = fit(&observation).unwrap();
            elapsed.push(start.elapsed().as_secs_f64());
            last = Some(result);
        }
        let result = last.unwrap();
        elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());

        eprintln!(
            "n={:<9} events={:<9} median={:.6}s  mu={:.10} alpha={:.10} beta={:.10} \
             nll={:.6} iters={} grad={:.3e} converged={}",
            nominal,
            times.len(),
            elapsed[TIMED / 2],
            result.parameters.baseline(),
            result.parameters.excitation(),
            result.parameters.decay(),
            result.negative_log_likelihood,
            result.iterations,
            result.gradient_norm,
            result.converged,
        );

        records.push(format!(
            r#"    {{
      "nominal_n": {nominal},
      "events": {},
      "horizon": {horizon:?},
      "seconds_median": {:?},
      "seconds_min": {:?},
      "seconds_max": {:?},
      "seconds_all": [{}],
      "baseline": {:?},
      "excitation": {:?},
      "decay": {:?},
      "negative_log_likelihood": {:?},
      "iterations": {},
      "gradient_norm": {:?},
      "converged": {}
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
            result.parameters.baseline(),
            result.parameters.excitation(),
            result.parameters.decay(),
            result.negative_log_likelihood,
            result.iterations,
            result.gradient_norm,
            result.converged,
        ));
    }

    let json = format!(
        "{{\n  \"side\": \"hawkes\",\n  \"warmup\": {WARMUP},\n  \"timed\": {TIMED},\n  \
         \"seed\": {SEED},\n  \"truth\": {{\"baseline\": {BASELINE:?}, \"excitation\": \
         {EXCITATION:?}, \"decay\": {DECAY:?}}},\n  \"runs\": [\n{}\n  ]\n}}\n",
        records.join(",\n")
    );
    fs::write(format!("{out_dir}/hawkes.json"), json).unwrap();
}
