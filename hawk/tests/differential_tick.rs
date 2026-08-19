//! Differential test harness against `tick` (CLAUDE.md §3, oracle 5).
//!
//! Loads every fixture in `tests/fixtures/`, which were generated inside the pinned
//! `tick` image (`benchmarks/docker/`), and replays each recorded parameter point
//! through `hawk`'s negative log-likelihood.
//!
//! # M0 status
//!
//! `hawk` has no log-likelihood yet, so [`stub_negative_log_likelihood`] plays the
//! recorded value back. That makes the comparison tautological on purpose: what is
//! under test in M0 is the *harness* — fixture discovery, parsing, the sweep over
//! every scenario and every parameter point, the tolerance, and the failure
//! message. The stub is deleted in M1 and replaced by the real implementation.
//!
//! [`fixtures_are_internally_consistent`] is not tautological: it validates the
//! committed corpus against facts that do not come from `hawk`.
//!
//! # Sabotage (CLAUDE.md §3)
//!
//! Confirmed to detect failure before being trusted. Perturbing
//! [`stub_negative_log_likelihood`] by `+ 1e-6` — a thousand times the tolerance,
//! but far too small to notice by eye in a fixture — turned
//! `differential_against_tick` red on all 24 parameter points. Transposing the
//! adjacency matrix in `fixture_evaluation_coeffs_use_ticks_layout` turned that test
//! red on both asymmetric fixtures and, as expected, left the symmetric one green.
//! Recorded in `docs/verification-log.md`.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

/// Absolute agreement required between `hawk` and `tick`.
///
/// CLAUDE.md §3 oracle 5 specifies 1e-9. `tick` returns `f64` and the fixtures
/// record its shortest round-trip decimal repr, so no precision is lost in
/// transport; 1e-9 leaves roughly six orders of magnitude of headroom over f64
/// round-off on values of order 1, which is where these losses sit.
const LOG_LIKELIHOOD_TOLERANCE: f64 = 1e-9;

/// Tolerance for checking that a fixture's flat `coeffs` vector agrees with its
/// `baseline` and `adjacency` fields. Both are written by the same generator from
/// the same `f64`s, so they must agree to the last bit; this is not a numerical
/// tolerance but a guard against a malformed fixture.
const EXACT: f64 = 0.0;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    name: String,
    n_nodes: usize,
    decay: f64,
    end_time: f64,
    baseline: Vec<f64>,
    adjacency: Vec<Vec<f64>>,
    spectral_radius: f64,
    n_jumps: usize,
    /// `events[j]` holds the timestamps of component `j`.
    events: Vec<Vec<f64>>,
    evaluations: Vec<Evaluation>,
}

#[derive(Debug, Deserialize)]
struct Evaluation {
    label: String,
    baseline: Vec<f64>,
    adjacency: Vec<Vec<f64>>,
    /// `tick`'s flat parameter vector: baseline block then adjacency raveled in C
    /// order. See `docs/derivations/conventions.md` C6.
    coeffs: Vec<f64>,
    /// `ModelHawkesExpKernLogLik.loss` at `coeffs`. This is *not* the negative
    /// log-likelihood: it is normalized by `n_jumps` and carries a `-D*T` offset.
    /// See conventions.md C7 and `docs/open-questions.md` OQ-8.
    tick_loss: f64,
    tick_grad: Vec<f64>,
}

fn fixture_dir() -> PathBuf {
    // Fixtures live at the repository root (CLAUDE.md §7), one level above this
    // crate, because they are shared reference data rather than crate-private.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures")
}

fn load_fixtures() -> Vec<Fixture> {
    let dir = fixture_dir();
    let entries = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read fixture directory {}: {e}", dir.display()));

    let mut paths: Vec<PathBuf> = entries
        .map(|entry| entry.expect("cannot read fixture directory entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    // Sorted so failures are reported in a stable order across platforms.
    paths.sort();

    assert!(
        !paths.is_empty(),
        "no fixtures found in {}. Regenerate them with the pinned tick image; see \
         benchmarks/docker/README.md",
        dir.display()
    );

    paths
        .iter()
        .map(|path| {
            let text = fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
        })
        .collect()
}

/// Stand-in for `hawk`'s negative log-likelihood. **Deleted in M1.**
///
/// Plays the oracle value back verbatim. Its only job is to let the harness run
/// end to end and to go red when perturbed.
fn stub_negative_log_likelihood(_fixture: &Fixture, evaluation: &Evaluation) -> f64 {
    evaluation.tick_loss
}

#[test]
fn differential_against_tick() {
    let fixtures = load_fixtures();
    let mut points = 0;

    for fixture in &fixtures {
        for evaluation in &fixture.evaluations {
            let actual = stub_negative_log_likelihood(fixture, evaluation);
            let discrepancy = (actual - evaluation.tick_loss).abs();
            assert!(
                discrepancy <= LOG_LIKELIHOOD_TOLERANCE,
                "{}/{}: hawk gave {actual:?}, tick gave {:?} \
                 (|difference| {discrepancy:?} > tolerance {LOG_LIKELIHOOD_TOLERANCE:?})",
                fixture.name,
                evaluation.label,
                evaluation.tick_loss,
            );
            points += 1;
        }
    }

    // Guards against the harness silently passing because it swept nothing.
    assert_eq!(
        points,
        fixtures.iter().map(|f| f.evaluations.len()).sum::<usize>()
    );
    assert!(
        points >= 20,
        "expected a broad sweep, only compared {points} points"
    );
}

#[test]
fn fixtures_are_internally_consistent() {
    let fixtures = load_fixtures();

    assert!(
        fixtures.len() >= 5,
        "CLAUDE.md §3 wants a corpus, found only {} fixtures",
        fixtures.len()
    );
    assert!(
        fixtures.iter().any(|f| f.n_nodes == 1),
        "corpus has no univariate fixture"
    );
    assert!(
        fixtures.iter().any(|f| f.n_nodes > 1),
        "corpus has no multivariate fixture"
    );

    for fixture in &fixtures {
        let name = &fixture.name;
        assert_eq!(
            fixture.schema_version, 1,
            "{name}: unexpected schema version"
        );
        assert!(fixture.decay > 0.0, "{name}: decay must be positive");
        assert!(fixture.end_time > 0.0, "{name}: end_time must be positive");

        assert_eq!(
            fixture.baseline.len(),
            fixture.n_nodes,
            "{name}: baseline length"
        );
        assert_eq!(
            fixture.events.len(),
            fixture.n_nodes,
            "{name}: events length"
        );
        assert_eq!(
            fixture.adjacency.len(),
            fixture.n_nodes,
            "{name}: adjacency rows"
        );
        for row in &fixture.adjacency {
            assert_eq!(
                row.len(),
                fixture.n_nodes,
                "{name}: adjacency is not square"
            );
        }

        assert!(
            fixture.baseline.iter().all(|&mu| mu > 0.0),
            "{name}: every baseline must be positive"
        );
        assert!(
            fixture.adjacency.iter().flatten().all(|&a| a >= 0.0),
            "{name}: adjacency must be non-negative"
        );

        // Under tick's kernel convention the adjacency matrix *is* the branching
        // matrix, so stationarity is spectral_radius < 1
        // (docs/derivations/conventions.md C2). A non-stationary fixture would be a
        // generator bug: these are simulated on a finite horizon and must terminate.
        assert!(
            fixture.spectral_radius < 1.0,
            "{name}: spectral radius {} is not stationary",
            fixture.spectral_radius
        );

        let counted: usize = fixture.events.iter().map(Vec::len).sum();
        assert_eq!(
            counted, fixture.n_jumps,
            "{name}: n_jumps disagrees with events"
        );
        assert!(fixture.n_jumps > 0, "{name}: no events");

        for (component, timestamps) in fixture.events.iter().enumerate() {
            assert!(
                timestamps.windows(2).all(|w| w[0] <= w[1]),
                "{name}: component {component} timestamps are not sorted"
            );
            assert!(
                timestamps
                    .iter()
                    .all(|&t| t >= 0.0 && t <= fixture.end_time),
                "{name}: component {component} has a timestamp outside [0, end_time]"
            );
        }

        assert!(
            !fixture.evaluations.is_empty(),
            "{name}: no evaluation points"
        );
        assert!(
            fixture.evaluations.iter().any(|e| e.label == "truth"),
            "{name}: no evaluation at the true parameters"
        );
        for evaluation in &fixture.evaluations {
            assert!(
                evaluation.tick_loss.is_finite(),
                "{name}/{}: tick_loss is not finite",
                evaluation.label
            );
            assert_eq!(
                evaluation.tick_grad.len(),
                fixture.n_nodes + fixture.n_nodes * fixture.n_nodes,
                "{name}/{}: gradient length disagrees with tick's coefficient layout",
                evaluation.label
            );
            assert!(
                evaluation.tick_grad.iter().all(|g| g.is_finite()),
                "{name}/{}: tick_grad has a non-finite entry",
                evaluation.label
            );
        }
    }
}

/// Pins the flat coefficient layout the fixtures were written with.
///
/// If `hawk` ever reads `coeffs` assuming a different order — column-major, or
/// adjacency before baseline — this is what catches it. The check is meaningful
/// only on asymmetric fixtures, so it asserts that some exist.
#[test]
fn fixture_evaluation_coeffs_use_ticks_layout() {
    let fixtures = load_fixtures();
    let mut asymmetric_seen = 0;

    for fixture in &fixtures {
        for evaluation in &fixture.evaluations {
            let mut expected = evaluation.baseline.clone();
            for row in &evaluation.adjacency {
                expected.extend_from_slice(row);
            }
            assert_eq!(
                evaluation.coeffs.len(),
                expected.len(),
                "{}/{}: coeffs length",
                fixture.name,
                evaluation.label
            );
            for (index, (&actual, &want)) in
                evaluation.coeffs.iter().zip(expected.iter()).enumerate()
            {
                assert!(
                    (actual - want).abs() <= EXACT,
                    "{}/{}: coeffs[{index}] is {actual:?}, expected {want:?}. \
                     The adjacency block must be raveled row-major after the \
                     baseline block (conventions.md C6).",
                    fixture.name,
                    evaluation.label
                );
            }
        }

        let is_asymmetric = (0..fixture.n_nodes).any(|i| {
            (0..fixture.n_nodes).any(|j| fixture.adjacency[i][j] != fixture.adjacency[j][i])
        });
        if is_asymmetric {
            asymmetric_seen += 1;
        }
    }

    assert!(
        asymmetric_seen >= 2,
        "a transposed adjacency is only detectable on asymmetric fixtures; found \
         {asymmetric_seen}"
    );
}
