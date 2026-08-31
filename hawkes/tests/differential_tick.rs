//! Differential test harness against `tick` (CLAUDE.md §3, oracle 5).
//!
//! Loads every fixture in `tests/fixtures/`, which were generated inside the pinned
//! `tick` image (`benchmarks/docker/`), and replays each recorded parameter point
//! through `hawkes`'s negative log-likelihood.
//!
//! # What is compared
//!
//! `tick`'s `loss` is neither the negative log-likelihood nor the formula in its own
//! docstring. It is normalized by the total jump count and carries an offset. The
//! identity under test is
//!
//! ```text
//! hawk_nll == tick_loss * n_jumps + D*T
//! ```
//!
//! which is OQ-8, closed here (M1 Part B step 9). The M0 stub that played the
//! recorded value back is gone; this now runs `hawkes`'s own implementation.
//!
//! Every fixture is compared, at `d` in {1, 2, 3, 10}.
//!
//! `D` varying is not incidental. At a single `d` the offset `D*T` cannot be
//! distinguished from `3*T`, or from any other multiple that happens to coincide
//! there — a test that passes for one `D` is not testing the offset. The corpus
//! carries `D*T` values of 20, 200, 2000, 600, 800, 1500 and 3000, and this test
//! asserts that several distinct `D` are actually exercised.
//!
//! # Why this comes last
//!
//! `hawkes`'s likelihood is gated against a brute-force transcription of the definition
//! (`loglikelihood.rs`), which is itself validated against hand calculations and the
//! Poisson degenerate case (`reference_loglikelihood.rs`). Neither uses `tick`. Only
//! after that is `tick` brought in — otherwise using `hawkes` to decide what `tick`
//! computes, having used `tick` to decide what `hawkes` should compute, would be
//! circular.
//!
//! # Sabotage (CLAUDE.md §3)
//!
//! Confirmed to detect failure before being trusted, in M0 against the stub and again
//! in M1 against the real implementation. Recorded in `docs/verification-log.md`.

use std::fs;
use std::path::PathBuf;

use hawkes::multivariate;
use hawkes::univariate::{Observation, Parameters, negative_log_likelihood};
use serde::Deserialize;

mod common;
use common::multivariate_computation_scale as multivariate_scale;

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
    n_jumps: usize,
    /// True when the scenario contains exact ties. Tied scenarios are hand-built,
    /// not simulated: Ogata thinning draws continuous inter-arrival times and never
    /// produces two events at one instant.
    has_ties: bool,
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
        // `rust-nll.json` is a manifest of values hawkes computes, not a fixture; it is
        // kept current by `rust_nll_manifest.rs` and consumed by the Python bindings'
        // bit-identity test.
        .filter(|path| path.file_name().is_some_and(|name| name != "rust-nll.json"))
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

/// Absolute agreement required, converted from the relative gate below.
///
/// CLAUDE.md §3 oracle 5 specifies 1e-9. Applied to the **scale of the computation**
/// rather than to `|nll|`, for the reason given in
/// `docs/derivations/univariate_loglikelihood.md` §5: `nll` is a difference of large
/// terms and passes through zero, so a gate relative to it diverges on correct code
/// wherever they cancel.
///
/// This is looser than the 1e-12 used against the brute force, and deliberately so.
/// There the two sides are the same arithmetic in a different order. Here the value
/// has crossed a language boundary, been reduced by `tick`'s own summation order,
/// divided by `n_jumps`, printed as decimal, and multiplied back up.
const TICK_TOLERANCE: f64 = 1e-9;

#[test]
fn differential_against_tick() {
    let fixtures = load_fixtures();
    let mut compared = 0;
    let mut with_excitation = 0;
    let mut tied = 0;
    let mut dimensions = std::collections::BTreeSet::new();

    for fixture in &fixtures {
        let observation = multivariate::Observation::new(&fixture.events, fixture.end_time)
            .unwrap_or_else(|e| {
                panic!("{}: fixture violates the input contract: {e}", fixture.name)
            });

        for evaluation in &fixture.evaluations {
            // Row-major, `excitation[i*d + j]` = "j excites i", matching the fixture's
            // `adjacency[i][j]` (conventions.md C6).
            let excitation: Vec<f64> = evaluation.adjacency.iter().flatten().copied().collect();
            let parameters = multivariate::Parameters::new(
                evaluation.baseline.clone(),
                excitation,
                fixture.decay,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));

            let hawk_nll =
                multivariate::negative_log_likelihood(&parameters, &observation).unwrap();
            // OQ-8: tick's loss is the negative log-likelihood ratio against a
            // unit-rate Poisson, divided by n_jumps. Undoing both gives the plain
            // negative log-likelihood. `D` is the number of components.
            let from_tick = evaluation.tick_loss * (fixture.n_jumps as f64)
                + (fixture.n_nodes as f64) * fixture.end_time;

            let scale = multivariate_scale(&parameters, &observation);
            let discrepancy = (hawk_nll - from_tick).abs();
            assert!(
                discrepancy <= TICK_TOLERANCE * scale,
                "{}/{} (d={}): hawkes {hawk_nll:?} vs tick_loss*n_jumps + D*T \
                 {from_tick:?}, |difference| {discrepancy:e} > {TICK_TOLERANCE:e} \
                 * scale {scale:e}. D*T here is {:?}.",
                fixture.name,
                evaluation.label,
                fixture.n_nodes,
                (fixture.n_nodes as f64) * fixture.end_time,
            );

            compared += 1;
            dimensions.insert(fixture.n_nodes);
            if evaluation.adjacency.iter().flatten().any(|&a| a != 0.0) {
                with_excitation += 1;
            }
            if fixture.has_ties {
                tied += 1;
            }
        }
    }

    // Guards against the harness passing because it swept nothing, and against the
    // corpus losing the cases that make the comparison meaningful.
    assert!(compared >= 40, "only compared {compared} points");
    assert!(
        with_excitation >= 30,
        "only {with_excitation} points had a non-zero excitation; the offset is \
         trivially confirmed at alpha == 0 and that is what left OQ-8 open"
    );
    assert!(
        tied >= 8,
        "only {tied} tied points; ties are an independent witness"
    );
    // The offset is D*T. With one value of D it cannot be distinguished from a
    // constant multiple of T, so several are required.
    assert!(
        dimensions.len() >= 4,
        "compared only {} distinct dimensions ({dimensions:?}); D must vary or the \
         D*T offset is not what is being tested",
        dimensions.len()
    );
    assert!(
        dimensions.contains(&10),
        "the d = 10 fixture is missing; it is the widest separation of D in the corpus"
    );
}

/// The univariate path must agree with the multivariate one on the univariate
/// fixtures. `multivariate_equivalence.rs` proves that in general; this checks it on
/// the committed corpus, against `tick`-derived values rather than synthetic ones.
#[test]
fn univariate_path_agrees_on_the_univariate_fixtures() {
    let fixtures = load_fixtures();
    let mut checked = 0;
    for fixture in &fixtures {
        if fixture.n_nodes != 1 {
            continue;
        }
        let observation = Observation::new(&fixture.events[0], fixture.end_time).unwrap();
        for evaluation in &fixture.evaluations {
            let parameters = Parameters::new(
                evaluation.baseline[0],
                evaluation.adjacency[0][0],
                fixture.decay,
            )
            .unwrap();
            let univariate_value = negative_log_likelihood(&parameters, &observation);

            let multi_observation =
                multivariate::Observation::new(&fixture.events, fixture.end_time).unwrap();
            let multi_parameters = multivariate::Parameters::new(
                evaluation.baseline.clone(),
                vec![evaluation.adjacency[0][0]],
                fixture.decay,
            )
            .unwrap();
            let multivariate_value =
                multivariate::negative_log_likelihood(&multi_parameters, &multi_observation)
                    .unwrap();

            assert_eq!(
                univariate_value.to_bits(),
                multivariate_value.to_bits(),
                "{}/{}: univariate {univariate_value:?} vs multivariate \
                 {multivariate_value:?}",
                fixture.name,
                evaluation.label
            );
            checked += 1;
        }
    }
    assert!(checked >= 20, "only checked {checked} univariate points");
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
    // Tied fixtures are the only independent witness for the grouped recursion of
    // docs/derivations/univariate_loglikelihood.md §4.2. Without them, tie handling
    // is checked only against hawkes's own brute force, and both come from the same
    // derivation. `tick` resolves ties by time rather than by array index, so these
    // are expected to agree rather than to fail -- measured in
    // benchmarks/docker/tie_identity.py.
    assert!(
        fixtures.iter().filter(|f| f.has_ties).count() >= 3,
        "corpus needs tied fixtures to witness the grouped recursion independently"
    );
    assert!(
        fixtures.iter().any(|f| f.has_ties && f.n_nodes > 1),
        "corpus has no multivariate tied fixture; cross-component ties are what an \
         index-based implementation is most likely to get wrong"
    );

    for fixture in &fixtures {
        let name = &fixture.name;
        assert_eq!(
            fixture.schema_version, 3,
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
        // matrix, so stationarity is spectral radius < 1
        // (docs/derivations/conventions.md C2). A non-stationary fixture would be a
        // generator bug: the simulated ones run on a finite horizon and must
        // terminate, and the hand-built ones were chosen stationary.
        //
        // Computed here rather than read from the file. The fixture used to store it,
        // and that value came from LAPACK, whose kernel selection depends on the CPU
        // it runs on -- which made the d = 10 fixture pass and fail across CI runs of
        // identical content. `hawkes`'s routine is pure f64 arithmetic and
        // deterministic; `spectral_radius.rs` pins it, including on the reducible
        // matrices where the naive method is wrong.
        let excitation: Vec<f64> = fixture.adjacency.iter().flatten().copied().collect();
        let parameters =
            multivariate::Parameters::new(fixture.baseline.clone(), excitation, fixture.decay)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            parameters.is_stationary(),
            "{name}: branching-ratio spectral radius {} is not stationary",
            parameters.branching_ratio_spectral_radius()
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
            // Non-strict above: ties are admitted by the input contract
            // (conventions.md C8). Cross-check the declared flag so a fixture cannot
            // quietly gain or lose ties without the corpus assertions noticing.
            let component_has_ties = timestamps.windows(2).any(|w| w[0] == w[1]);
            assert!(
                !component_has_ties || fixture.has_ties,
                "{name}: component {component} has ties but has_ties is false"
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
/// If `hawkes` ever reads `coeffs` assuming a different order — column-major, or
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
