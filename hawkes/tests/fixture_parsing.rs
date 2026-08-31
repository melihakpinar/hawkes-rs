//! The fixture reader must recover the exact doubles the corpus was written from.
//!
//! The fixtures are emitted by CPython with shortest-round-trip float repr, so every
//! literal in them has exactly one nearest `f64` and reading it back should recover
//! that value bit for bit.
//!
//! `serde_json`'s default float parsing does **not** guarantee that. Its fast path is
//! off by one ulp on some inputs, including several in this corpus, which meant Rust
//! was computing on events up to one ulp away from the ones `tick` actually simulated.
//! The `float_roundtrip` feature fixes it and is enabled in the workspace manifest.
//!
//! Nothing caught this for three milestones. The `tick` differential compares to
//! `1e-9` relative, seven orders of magnitude above one ulp, so it passed throughout.
//! It surfaced only when M3 required the Python bindings to agree **bitwise** with
//! Rust, and the two sides turned out to be parsing the same file differently.
//!
//! # Sabotage
//!
//! Removing `features = ["float_roundtrip"]` from `serde_json` turns this red on the
//! literals below. Recorded in `docs/verification-log.md`.

/// Literals taken from `tests/fixtures/bivariate_symmetric.json`, with the nearest
/// `f64` determined by exact decimal comparison rather than by another parser: for
/// `12.992302737526387` the chosen double is `9.08e-17` from the true value and its
/// neighbours are `1.69e-15` and `1.87e-15` away, so the answer is not in doubt.
const CORRECTLY_ROUNDED: [(&str, u64); 3] = [
    ("12.992302737526387", 0x4029_fc0f_1aba_d070),
    ("100.68066598854051", 0x4059_2b90_0814_11fc),
    ("104.38219480240177", 0x405a_1875_e130_4113),
];

#[test]
fn serde_json_parses_fixture_floats_to_the_nearest_double() {
    for (literal, expected) in CORRECTLY_ROUNDED {
        let parsed: f64 = serde_json::from_str(literal).expect("valid JSON number");
        assert_eq!(
            parsed.to_bits(),
            expected,
            "{literal} parsed to 0x{:016x} ({parsed:?}), nearest double is \
             0x{expected:016x} ({:?}). serde_json's `float_roundtrip` feature is \
             probably not enabled; without it the fixture corpus is read one ulp away \
             from the values tick produced.",
            parsed.to_bits(),
            f64::from_bits(expected)
        );
    }
}

/// The same check through the type the fixtures are actually read as, so the feature
/// cannot be enabled for a bare `f64` and lost for a struct field.
#[test]
fn fixture_shaped_parsing_is_also_exact() {
    #[derive(serde::Deserialize)]
    struct Events {
        events: Vec<Vec<f64>>,
    }
    let json = format!(
        r#"{{"events": [[{}, {}], [{}]]}}"#,
        CORRECTLY_ROUNDED[0].0, CORRECTLY_ROUNDED[1].0, CORRECTLY_ROUNDED[2].0
    );
    let parsed: Events = serde_json::from_str(&json).expect("valid");
    assert_eq!(parsed.events[0][0].to_bits(), CORRECTLY_ROUNDED[0].1);
    assert_eq!(parsed.events[0][1].to_bits(), CORRECTLY_ROUNDED[1].1);
    assert_eq!(parsed.events[1][0].to_bits(), CORRECTLY_ROUNDED[2].1);
}

// ---------------------------------------------------------------------------------
// Permanent guards. The three literals above are a snapshot of one corpus; these two
// tests hold whatever the corpus becomes, and hold for the loader rather than for a
// bare `f64`.
// ---------------------------------------------------------------------------------

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::PathBuf;

/// The shape the fixtures are actually read as — a struct field, inside a nested
/// `Vec`. `float_roundtrip` could in principle be lost for this path while a bare
/// `f64` still parsed correctly, so the round-trip is exercised through it.
#[derive(serde::Serialize, serde::Deserialize)]
struct EventsFixture {
    events: Vec<Vec<f64>>,
}

/// Writes `values` as a fixture, reads it back, and requires every bit to survive.
///
/// The oracle is the **input**: no parser supplies the expected answer, so this is
/// independent of the code under test in the sense CLAUDE.md §3 requires.
#[track_caller]
fn assert_round_trips(values: &[f64], context: &str) {
    let written = serde_json::to_string(&EventsFixture {
        events: vec![values.to_vec()],
    })
    .expect("f64s are serializable");
    let read: EventsFixture = serde_json::from_str(&written).expect("valid JSON");
    assert_eq!(
        read.events[0].len(),
        values.len(),
        "{context}: length changed"
    );
    for (index, (&original, &recovered)) in values.iter().zip(&read.events[0]).enumerate() {
        assert_eq!(
            original.to_bits(),
            recovered.to_bits(),
            "{context}: value {index} was written from 0x{:016x} ({original:?}) and read \
             back as 0x{:016x} ({recovered:?}), a difference of {} ulp. serde_json's \
             `float_roundtrip` feature is probably not enabled.",
            original.to_bits(),
            recovered.to_bits(),
            original.to_bits().abs_diff(recovered.to_bits())
        );
    }
}

/// Values chosen to sit where float parsers go wrong, rather than where they are easy.
///
/// A fixture holds event times and parameters, so most of these will never appear in
/// one. That is deliberate: the test is here to detect a parser regression, and the
/// regressions live at the extremes even when the corpus does not.
#[test]
fn hard_values_survive_the_fixture_round_trip() {
    let cases: [(&str, f64); 12] = [
        ("smallest positive subnormal", f64::from_bits(1)),
        ("largest subnormal", f64::from_bits(0x000f_ffff_ffff_ffff)),
        ("smallest normal", f64::MIN_POSITIVE),
        (
            "one ulp above the smallest normal",
            f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1),
        ),
        ("largest finite", f64::MAX),
        ("a 17-significant-digit decimal", 0.1),
        ("one third", 1.0 / 3.0),
        (
            "just below a power of two",
            f64::from_bits(0x3ff0_0000_0000_0000 - 1),
        ),
        (
            "just above a power of two",
            f64::from_bits(0x3ff0_0000_0000_0000 + 1),
        ),
        (
            "a corpus literal that the fast path got wrong",
            f64::from_bits(0x4029_fc0f_1aba_d070),
        ),
        ("another", f64::from_bits(0x4059_2b90_0814_11fc)),
        ("a third", f64::from_bits(0x405a_1875_e130_4113)),
    ];
    for (name, value) in cases {
        assert_round_trips(&[value], name);
    }
    // Together, so a defect that depends on position within an array is also covered.
    let all: Vec<f64> = cases.iter().map(|&(_, v)| v).collect();
    assert_round_trips(&all, "all hard values in one array");
}

/// A fixed seed exercises one draw; a defect whose trigger is a particular mantissa is
/// invisible to it (CLAUDE.md §3). This sweeps the whole finite `f64` range.
///
/// Random *bit patterns* rather than random magnitudes: sampling `f64` uniformly on
/// `[0, 1)` would never produce a subnormal, a huge exponent, or a mantissa with a long
/// decimal expansion, which is exactly the population that matters here.
#[test]
fn a_sweep_of_random_bit_patterns_survives_the_round_trip() {
    const SEED: u64 = 0x5eed_f10a_7c0d_e42a;
    const DRAWS: usize = 20_000;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut batch = Vec::with_capacity(DRAWS);
    while batch.len() < DRAWS {
        let candidate = f64::from_bits(rng.random::<u64>());
        if candidate.is_finite() {
            batch.push(candidate);
        }
    }
    assert_round_trips(&batch, "random bit patterns");
}

/// Every float literal in every committed fixture must parse to the same double under
/// `serde_json` as under the standard library.
///
/// `str::parse::<f64>` is correctly rounded and is a separate implementation in a
/// separate crate, so this is a differential test rather than a self-check. It covers
/// all thirteen of the originally affected points without hard-coding them, and keeps
/// covering the corpus as scenarios are added — which the three pinned literals above
/// cannot do.
#[test]
fn every_fixture_literal_parses_to_the_same_double_as_the_standard_library() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures");
    let mut files = 0usize;
    let mut literals = 0usize;
    for entry in std::fs::read_dir(&dir).expect("fixture directory") {
        let path = entry.expect("readable entry").path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        files += 1;
        let text = std::fs::read_to_string(&path).expect("readable fixture");
        for literal in float_literals(&text) {
            literals += 1;
            let via_serde: f64 = serde_json::from_str(literal).expect("valid JSON number");
            let via_std: f64 = literal.parse().expect("valid Rust float");
            assert_eq!(
                via_serde.to_bits(),
                via_std.to_bits(),
                "{}: {literal} parsed to 0x{:016x} by serde_json and 0x{:016x} by the \
                 standard library. serde_json's `float_roundtrip` feature is probably \
                 not enabled; without it the corpus is read away from the values tick \
                 produced.",
                path.display(),
                via_serde.to_bits(),
                via_std.to_bits()
            );
        }
    }
    assert!(
        files >= 11,
        "expected the committed corpus, found {files} files"
    );
    assert!(
        literals >= 7_000,
        "expected thousands of literals, scanned {literals}; the scanner is probably \
         not finding them and the test is passing vacuously"
    );
}

/// Number tokens in a JSON document, skipping anything inside a string.
///
/// Deliberately does not use `serde_json` to find them: that is the code under test,
/// and a parser that mis-reads a number could just as well mis-report where it was.
fn float_literals(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut in_string = false;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if c == b'-' || c.is_ascii_digit() {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'-' | b'+' | b'.' | b'e' | b'E'))
            {
                i += 1;
            }
            let token = &text[start..i];
            if token.contains('.') || token.contains('e') || token.contains('E') {
                out.push(token);
            }
            continue;
        }
        i += 1;
    }
    out
}
