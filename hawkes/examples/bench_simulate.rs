//! `hawkes` side of the M4 simulate benchmark. See `benchmarks/README.md`.
//!
//! Usage: `bench_simulate <out_dir> <d> <n,n,...>`

use std::fs;
use std::time::Instant;

use hawkes::multivariate::{Parameters, simulate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 20_260_831;
const DECAY: f64 = 1.0;
const WARMUP: usize = 1;
const TIMED: usize = 5;
const SINGLE_RUN_BUDGET: f64 = 600.0;
const CELL_BUDGET: f64 = 1800.0;

fn truth(d: usize) -> Parameters {
    Parameters::new(vec![0.5 / d as f64; d], vec![0.6 / d as f64; d * d], DECAY)
        .expect("valid parameters")
}

fn json_array(values: &[f64]) -> String {
    let parts: Vec<String> = values.iter().map(|v| format!("{v:?}")).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args
        .next()
        .expect("usage: bench_simulate <dir> <d> <n,n,...>");
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

    let mut records = Vec::new();
    for nominal in grid {
        let horizon = nominal as f64 / rate;

        // The seed is advanced per repetition so no run reuses another's stream; the
        // realized count therefore varies between repetitions and is reported as a
        // median alongside the time.
        for repetition in 0..WARMUP {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED + repetition as u64);
            std::hint::black_box(simulate(&truth, horizon, &mut rng).unwrap());
        }

        let cell_start = Instant::now();
        let mut elapsed = Vec::with_capacity(TIMED);
        let mut counts = Vec::with_capacity(TIMED);
        let mut aborted = None;
        for repetition in 0..TIMED {
            let mut rng = ChaCha8Rng::seed_from_u64(SEED + 100 + repetition as u64);
            let start = Instant::now();
            let events = simulate(&truth, horizon, &mut rng).unwrap();
            let seconds = start.elapsed().as_secs_f64();
            counts.push(events.iter().map(|c| c.len()).sum::<usize>());
            if seconds > SINGLE_RUN_BUDGET {
                aborted = Some("single run exceeded 600 s");
                break;
            }
            elapsed.push(seconds);
            if cell_start.elapsed().as_secs_f64() > CELL_BUDGET {
                aborted = Some("cell exceeded 1800 s");
                break;
            }
        }

        if let Some(reason) = aborted {
            eprintln!("d={d} n={nominal} ABORTED: {reason}");
            records.push(format!(
                r#"{{"dimension":{d},"nominal_n":{nominal},"horizon":{horizon:?},"completed":false,"abort_reason":"{reason}"}}"#
            ));
            continue;
        }

        elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());
        counts.sort_unstable();
        eprintln!(
            "d={d} n={nominal} median={:.6}s events_median={}",
            elapsed[TIMED / 2],
            counts[TIMED / 2]
        );
        records.push(format!(
            r#"{{"dimension":{d},"nominal_n":{nominal},"horizon":{horizon:?},"completed":true,"seconds_median":{:?},"seconds_min":{:?},"seconds_max":{:?},"seconds_all":{},"events_median":{},"events_all":{:?}}}"#,
            elapsed[TIMED / 2],
            elapsed[0],
            elapsed[TIMED - 1],
            json_array(&elapsed),
            counts[TIMED / 2],
            counts
        ));
    }

    let body = format!(
        r#"{{"library":"hawkes","benchmark":"simulate","dimension":{d},"seed":{SEED},"warmup":{WARMUP},"timed":{TIMED},"true_baseline":{},"true_excitation":{},"true_decay":{DECAY:?},"runs":[{}]}}"#,
        json_array(truth.baseline()),
        json_array(truth.excitation()),
        records.join(",")
    );
    fs::write(format!("{out_dir}/hawk_simulate_d{d}.json"), body + "\n").unwrap();
}
