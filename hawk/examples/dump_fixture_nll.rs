//! Writes the negative log-likelihood `hawk` computes for every fixture and parameter
//! point, as an exact bit pattern.
//!
//! This is what the Python bindings are compared against in
//! `hawk-python/tests/test_bit_identity.py`. Both sides call the same Rust function, so
//! a difference can only come from the boundary — a cast, a copy that reorders, a
//! reinterpreted stride. Bit patterns rather than decimal, because a decimal round trip
//! is itself a conversion and would hide exactly what is being looked for.
//!
//! # Why this is not committed
//!
//! Because the value is **platform-dependent**, and a committed artifact would be
//! asserting a cross-platform reproducibility that does not hold.
//!
//! It was committed at first, and CI failed on it: the manifest generated on macOS
//! aarch64 recorded `0x4086d455b0fc6646` for one point and Linux x86_64 computed
//! `0x4086d455b0fc6645`.
//!
//! The cause is not architecture as such. IEEE-754 requires `+`, `-`, `*` and `/` to be
//! correctly rounded, so those are identical everywhere; it requires nothing of the
//! sort for `exp`, `ln` and `exp_m1`, which come from the platform's libm. Apple's and
//! glibc's differ in the last bits, and this computation is built from them. (FMA
//! contraction was checked first and ruled out: `rustc -O` emits separate `fmul` and
//! `fadd` on aarch64, not `fmadd`.)
//!
//! So the manifest is generated next to the tests that consume it, on the machine that
//! will consume it, and lives in `target/`. A stale manifest fails the comparison
//! loudly rather than passing quietly, which is the safe direction.
//!
//! Usage: `cargo run --example dump_fixture_nll [output path]`, defaulting to
//! `target/rust-nll.json`.

use std::fs;
use std::path::PathBuf;

use hawk::multivariate;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    n_nodes: usize,
    decay: f64,
    end_time: f64,
    events: Vec<Vec<f64>>,
    evaluations: Vec<Evaluation>,
}

#[derive(Deserialize)]
struct Evaluation {
    label: String,
    baseline: Vec<f64>,
    adjacency: Vec<Vec<f64>>,
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures");
    let output = std::env::args().nth(1).map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/rust-nll.json"),
        PathBuf::from,
    );
    let mut paths: Vec<PathBuf> = fs::read_dir(&root)
        .expect("fixture directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter(|p| p.file_name().is_some_and(|n| n != "rust-nll.json"))
        .collect();
    paths.sort();

    let mut entries = Vec::new();
    for path in &paths {
        let fixture: Fixture =
            serde_json::from_str(&fs::read_to_string(path).expect("read")).expect("parse");
        let observation = multivariate::Observation::new(&fixture.events, fixture.end_time)
            .expect("fixture satisfies the input contract");
        for evaluation in &fixture.evaluations {
            let excitation: Vec<f64> = evaluation.adjacency.iter().flatten().copied().collect();
            let parameters = multivariate::Parameters::new(
                evaluation.baseline.clone(),
                excitation,
                fixture.decay,
            )
            .expect("fixture parameters are valid");
            let value = multivariate::negative_log_likelihood(&parameters, &observation).unwrap();
            entries.push(format!(
                "    {{\n      \"fixture\": {:?},\n      \"label\": {:?},\n      \
                 \"n_nodes\": {},\n      \"negative_log_likelihood_bits\": \"0x{:016x}\",\n      \
                 \"negative_log_likelihood\": {:?}\n    }}",
                fixture.name,
                evaluation.label,
                fixture.n_nodes,
                value.to_bits(),
                value,
            ));
        }
    }

    let json = format!(
        "{{\n  \"what\": \"negative log-likelihood computed by hawk in Rust, as exact \
         bit patterns\",\n  \"generated_by\": \"cargo run --example dump_fixture_nll\",\n  \
         \"kept_current_by\": \"hawk/tests/rust_nll_manifest.rs\",\n  \"entries\": [\n{}\n  ]\n}}\n",
        entries.join(",\n")
    );
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    fs::write(&output, json).expect("write");
    eprintln!("wrote {} entries to {}", entries.len(), output.display());
}
