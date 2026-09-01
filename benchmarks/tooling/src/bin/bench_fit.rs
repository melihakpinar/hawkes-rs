//! `hawkes` side of the M4 fit benchmarks. See `benchmarks/README.md`.
//!
//! Generates the shared data for one dimension, writes it out for the `tick` side, and
//! times the fit under the protocol in §4. Enforces the §4.1 budget.
//!
//! Usage: `bench_fit <out_dir> <d> <n,n,...>`

use std::fs;
use std::io::{BufWriter, Write};
use std::time::Instant;

use hawkes::multivariate::{Observation, Parameters, fit, simulate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_831;
const DECAY: f64 = 1.0;
const WARMUP: usize = 1;
const TIMED: usize = 5;
/// §4.1. Same numbers for both libraries.
const SINGLE_RUN_BUDGET: f64 = 600.0;
const CELL_BUDGET: f64 = 1800.0;

/// §3: `mu_i = 0.5/d`, `alpha_ij = 0.6/d`, so the spectral radius is 0.6 at every `d`.
fn truth(d: usize) -> Parameters {
    Parameters::new(vec![0.5 / d as f64; d], vec![0.6 / d as f64; d * d], DECAY)
        .expect("valid parameters")
}

fn write_events(path: &str, horizon: f64, events: &[Vec<f64>]) {
    let file = fs::File::create(path).expect("create events file");
    let mut out = BufWriter::new(file);
    // `{:?}` on f64 is shortest-round-trip, so Python's float() reads back the
    // identical double. Same guarantee the fixture corpus relies on.
    writeln!(out, "{horizon:?}").unwrap();
    writeln!(out, "{}", events.len()).unwrap();
    for component in events {
        writeln!(out, "{}", component.len()).unwrap();
        for t in component {
            writeln!(out, "{t:?}").unwrap();
        }
    }
}

fn json_array(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
    format!("[{}]", parts.join(","))
}

/// One file per grid point, written as soon as that point finishes, so a cell killed
/// at the §4.1 budget loses only the cell in flight rather than the whole dimension.
fn write_point(out_dir: &str, d: usize, nominal: usize, truth: &Parameters, run: String) {
    let body = format!(
        r#"{{"library":"hawkes","dimension":{d},"seed":{SEED},"warmup":{WARMUP},"timed":{TIMED},"true_baseline":{},"true_excitation":{},"true_decay":{DECAY:?},"run":{run}}}"#,
        json_array(truth.baseline()),
        json_array(truth.excitation()),
    );
    fs::write(format!("{out_dir}/hawk_d{d}_n{nominal}.json"), body + "\n").unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args.next().expect("usage: bench_fit <dir> <d> <n,n,...>");
    let d: usize = args.next().expect("dimension").parse().unwrap();
    let grid: Vec<usize> = args
        .next()
        .expect("event counts")
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    fs::create_dir_all(&out_dir).unwrap();

    let truth = truth(d);
    let rate: f64 = truth
        .stationary_mean_intensity()
        .expect("stationary")
        .iter()
        .sum();

    for nominal in grid {
        let horizon = nominal as f64 / rate;
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let events = simulate(&truth, horizon, &mut rng).unwrap();
        let realized: usize = events.iter().map(|c| c.len()).sum();
        write_events(
            &format!("{out_dir}/events_d{d}_n{nominal}.txt"),
            horizon,
            &events,
        );

        let observation = Observation::new(&events, horizon).unwrap();

        let cell_start = Instant::now();
        let mut aborted = None;
        for _ in 0..WARMUP {
            std::hint::black_box(fit(&observation).unwrap());
            if cell_start.elapsed().as_secs_f64() > CELL_BUDGET {
                aborted = Some("cell budget exceeded during warmup");
            }
        }
        let mut elapsed = Vec::with_capacity(TIMED);
        let mut last = None;
        while aborted.is_none() && elapsed.len() < TIMED {
            let start = Instant::now();
            let result = fit(&observation).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            if seconds > SINGLE_RUN_BUDGET {
                aborted = Some("single run exceeded 600 s");
                break;
            }
            elapsed.push(seconds);
            last = Some(result);
            if cell_start.elapsed().as_secs_f64() > CELL_BUDGET {
                aborted = Some("cell exceeded 1800 s");
                break;
            }
        }

        if let Some(reason) = aborted {
            eprintln!("d={d} n={nominal} events={realized} ABORTED: {reason}");
            write_point(
                &out_dir,
                d,
                nominal,
                &truth,
                format!(
                    r#"{{"dimension":{d},"nominal_n":{nominal},"events":{realized},"horizon":{horizon:?},"completed":false,"abort_reason":"{reason}"}}"#
                ),
            );
            continue;
        }

        elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let result = last.unwrap();
        let spectral = result.parameters.branching_ratio_spectral_radius();
        eprintln!(
            "d={d} n={nominal} events={realized} median={:.6}s nll={:.6} iters={} \
             grad={:.3e} converged={} rho={:.6}",
            elapsed[TIMED / 2],
            result.negative_log_likelihood,
            result.iterations,
            result.gradient_norm,
            result.converged,
            spectral
        );
        write_point(
            &out_dir,
            d,
            nominal,
            &truth,
            format!(
                r#"{{"dimension":{d},"nominal_n":{nominal},"events":{realized},"horizon":{horizon:?},"completed":true,"seconds_median":{:?},"seconds_min":{:?},"seconds_max":{:?},"seconds_all":{},"negative_log_likelihood":{:?},"iterations":{},"gradient_norm":{:?},"converged":{},"objective_evaluations":{},"gradient_evaluations":{},"spectral_radius":{:?},"baseline":{},"excitation":{},"decay":{:?}}}"#,
                elapsed[TIMED / 2],
                elapsed[0],
                elapsed[TIMED - 1],
                json_array(&elapsed),
                result.negative_log_likelihood,
                result.iterations,
                result.gradient_norm,
                result.converged,
                result.objective_evaluations,
                result.gradient_evaluations,
                spectral,
                json_array(result.parameters.baseline()),
                json_array(result.parameters.excitation()),
                result.parameters.decay(),
            ),
        );
    }
}
