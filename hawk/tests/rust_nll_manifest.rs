//! Keeps `tests/fixtures/rust-nll.json` current.
//!
//! That file records the negative log-likelihood `hawk` computes for every fixture and
//! parameter point, as an exact bit pattern. The Python bindings are compared against
//! it (`hawk-python/tests/test_bit_identity.py`), which is only meaningful if it
//! describes the code as it stands.
//!
//! A committed artifact that nothing checks is a comment that happens to be machine
//! readable. This test is what makes it evidence.
//!
//! # Sabotage
//!
//! Perturbing one entry's bit pattern by a single bit turned this red, naming the
//! fixture, the label and both patterns. Recorded in `docs/verification-log.md`.

use std::fs;
use std::path::PathBuf;

use hawk::multivariate;
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    fixture: String,
    label: String,
    negative_log_likelihood_bits: String,
}

#[derive(Deserialize)]
struct Fixture {
    name: String,
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

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures")
}

#[test]
fn the_committed_manifest_matches_what_hawk_computes() {
    let root = fixture_dir();
    let manifest: Manifest =
        serde_json::from_str(&fs::read_to_string(root.join("rust-nll.json")).expect(
            "rust-nll.json is missing; regenerate with `cargo run --example dump_fixture_nll`",
        ))
        .expect("rust-nll.json is not valid JSON");

    let mut paths: Vec<PathBuf> = fs::read_dir(&root)
        .expect("fixture directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter(|p| p.file_name().is_some_and(|n| n != "rust-nll.json"))
        .collect();
    paths.sort();

    let mut expected = Vec::new();
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
            expected.push((
                fixture.name.clone(),
                evaluation.label.clone(),
                multivariate::negative_log_likelihood(&parameters, &observation),
            ));
        }
    }

    assert_eq!(
        manifest.entries.len(),
        expected.len(),
        "the manifest has {} entries and the corpus has {}; regenerate with \
         `cargo run --example dump_fixture_nll`",
        manifest.entries.len(),
        expected.len()
    );
    assert!(
        expected.len() >= 40,
        "only {} points in the corpus",
        expected.len()
    );

    for (entry, (name, label, value)) in manifest.entries.iter().zip(&expected) {
        assert_eq!(&entry.fixture, name, "manifest is out of order");
        assert_eq!(&entry.label, label, "manifest is out of order");
        let recorded = u64::from_str_radix(
            entry
                .negative_log_likelihood_bits
                .strip_prefix("0x")
                .expect("bit pattern should be 0x-prefixed"),
            16,
        )
        .expect("bit pattern should be hexadecimal");
        assert_eq!(
            recorded,
            value.to_bits(),
            "{name}/{label}: manifest records 0x{recorded:016x} ({:?}), hawk now \
             computes 0x{:016x} ({value:?}). If the change is intended, regenerate \
             with `cargo run --example dump_fixture_nll`.",
            f64::from_bits(recorded),
            value.to_bits(),
        );
    }
}
