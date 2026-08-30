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
