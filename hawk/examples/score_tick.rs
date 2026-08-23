//! Scores `tick`'s returned parameters under `hawk`'s unpenalized objective, so both
//! answers sit in one common unit. See `docs/positioning-probe.md` §5.4.

use std::fs;

use hawk::univariate::{Observation, Parameters, negative_log_likelihood};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: score_tick <dir>");
    let spec = fs::read_to_string(format!("{dir}/tick_params.txt")).unwrap();
    for line in spec.lines().filter(|l| !l.trim().is_empty()) {
        // nominal_n baseline excitation decay
        let f: Vec<f64> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let nominal = f[0] as usize;
        let raw = fs::read_to_string(format!("{dir}/events_{nominal}.txt")).unwrap();
        let mut lines = raw.lines();
        let horizon: f64 = lines.next().unwrap().parse().unwrap();
        let times: Vec<f64> = lines.map(|l| l.parse().unwrap()).collect();
        let observation = Observation::new(&times, horizon).unwrap();
        let parameters = Parameters::new(f[1], f[2], f[3]).unwrap();
        println!(
            "{nominal} {:?}",
            negative_log_likelihood(&parameters, &observation)
        );
    }
}
