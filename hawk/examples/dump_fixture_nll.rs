//! Writes the negative log-likelihood `hawk` computes for every fixture and parameter
//! point, as an exact bit pattern, to `tests/fixtures/rust-nll.json`.
//!
//! This is what the Python bindings are compared against in
//! `hawk-python/tests/test_bit_identity.py`. Both sides call the same Rust function,
//! so a difference can only come from the boundary — a cast, a copy that reorders, a
//! reinterpreted stride. Bit patterns rather than decimal, because a decimal round
//! trip is itself a conversion and would hide exactly what is being looked for.
//!
//! `hawk/tests/rust_nll_manifest.rs` fails if the committed file is stale, so it
//! cannot drift away from the code it claims to describe.
//!
//! Regenerate with `cargo run --example dump_fixture_nll`.

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
            let value = multivariate::negative_log_likelihood(&parameters, &observation);
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
    let out = root.join("rust-nll.json");
    fs::write(&out, json).expect("write");
    eprintln!("wrote {} entries to {}", entries.len(), out.display());
}
