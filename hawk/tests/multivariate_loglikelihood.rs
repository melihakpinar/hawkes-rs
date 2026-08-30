//! The multivariate `O(n*d)` recursion against the `O(n^2)` definition (M2 Part B
//! step 8), and the gate that comparison uses.
//!
//! Primary correctness gate for the multivariate likelihood; uses no `tick`. The
//! reference is a direct transcription of the definition and is validated separately
//! against hand calculations here and in `reference_loglikelihood.rs`.
//!
//! # Sabotage
//!
//! Advancing the excitation state per event rather than per pooled distinct time — the
//! mistake `multivariate_loglikelihood.md` §4.1 is about — turned
//! `cross_component_ties_do_not_excite` and `agrees_with_brute_force_on_tied_input`
//! red while leaving every tie-free case green. Transposing the excitation index in
//! the intensity turned the asymmetric cases red and left the symmetric ones green.
//! Inflating `multivariate_computation_scale` turned the gate meta-tests red.
//! Recorded in `docs/verification-log.md`.

mod common;

use common::{
    Lcg, RECURSION_TOLERANCE, brute_force_multivariate_negative_log_likelihood as brute_force,
    multivariate_computation_scale as scale_of,
};
use hawk::multivariate::{Observation, Parameters, negative_log_likelihood};
use proptest::prelude::*;

/// Same bound and same reasoning as the univariate gate; see
/// `univariate_loglikelihood.md` §5.
const MAX_EVENTS_FOR_COMPARISON: usize = 50_000;

fn assert_agrees(parameters: &Parameters, observation: &Observation, context: &str) {
    let recursive = negative_log_likelihood(parameters, observation);
    let reference = brute_force(parameters, observation);
    let scale = scale_of(parameters, observation);
    let discrepancy = (recursive - reference).abs();
    assert!(
        discrepancy <= RECURSION_TOLERANCE * scale,
        "{context}: recursion {recursive:?} vs definition {reference:?}, \
         |difference| {discrepancy:e} > {RECURSION_TOLERANCE:e} * scale {scale:e}"
    );
}

fn parameters(baseline: Vec<f64>, excitation: Vec<f64>, decay: f64) -> Parameters {
    Parameters::new(baseline, excitation, decay).unwrap()
}

/// The counterexample from `multivariate_loglikelihood.md` §4.1, pinned.
///
/// Two components sharing the timestamp `2.5`. Neither event may excite the other.
/// Advancing the state per event instead gives `10.218429607528986` — wrong by 1.1%,
/// and wrong only because of the tie.
#[test]
fn cross_component_ties_do_not_excite() {
    let p = parameters(vec![0.2, 0.5], vec![0.1, 0.6, 0.05, 0.15], 1.2);
    let events = vec![vec![1.0, 2.5], vec![2.5, 4.0]];
    let observation = Observation::new(&events, 6.0).unwrap();

    let actual = negative_log_likelihood(&p, &observation);
    let expected = 10.329672183654555;
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs(),
        "got {actual:?}, hand calculation gives {expected:?}. Advancing the state \
         per event rather than per pooled distinct time gives 10.218429607528986."
    );
    assert_agrees(&p, &observation, "cross-component tie");

    // Break the tie and the per-event walk becomes correct too, which is why this
    // defect is invisible on simulated data.
    let untied = vec![vec![1.0, 2.5], vec![2.6, 4.0]];
    let untied_observation = Observation::new(&untied, 6.0).unwrap();
    let untied_value = negative_log_likelihood(&p, &untied_observation);
    assert!(
        (untied_value - 10.22398192492856).abs() <= 1e-12,
        "untied control gave {untied_value:?}"
    );
}

#[test]
fn agrees_with_brute_force_on_tied_input() {
    let p = parameters(
        vec![0.4, 0.5, 0.3],
        vec![0.1, 0.2, 0.0, 0.05, 0.15, 0.1, 0.2, 0.0, 0.1],
        1.3,
    );
    for events in [
        vec![vec![1.0, 2.0, 2.0], vec![2.0, 3.0], vec![2.0]],
        vec![vec![0.0, 0.0], vec![0.0], vec![0.0, 5.0, 5.0]],
        vec![vec![2.0; 5], vec![2.0; 3], vec![2.0]],
        vec![vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![], vec![1.0]],
    ] {
        let observation = Observation::new(&events, 6.0).unwrap();
        assert_agrees(&p, &observation, &format!("{events:?}"));
    }
}

#[test]
fn agrees_with_brute_force_on_degenerate_input() {
    let p = parameters(vec![0.9, 0.4], vec![0.3, 0.1, 0.0, 0.0], 1.7);
    for events in [
        vec![vec![], vec![]],
        vec![vec![2.5], vec![]],
        vec![vec![], vec![2.5]],
        vec![vec![0.0], vec![5.0]],
        vec![vec![0.0, 5.0], vec![0.0, 5.0]],
    ] {
        let observation = Observation::new(&events, 5.0).unwrap();
        assert_agrees(&p, &observation, &format!("{events:?}"));
    }
}

/// A zero row makes a component Poisson; a zero column makes it excite nothing.
/// Both are ordinary in `d` dimensions and neither needs a special case.
#[test]
fn agrees_with_brute_force_on_structural_zeros() {
    let events = vec![vec![0.5, 1.5, 3.0], vec![1.0, 2.0], vec![0.75, 2.5]];
    let observation = Observation::new(&events, 5.0).unwrap();
    for (label, excitation) in [
        (
            "zero row 1",
            vec![0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.2],
        ),
        (
            "zero column 2",
            vec![0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.1, 0.0],
        ),
        ("all zero", vec![0.0; 9]),
    ] {
        let p = parameters(vec![0.5, 0.6, 0.4], excitation, 1.1);
        assert_agrees(&p, &observation, label);
    }
}

#[test]
fn agrees_with_brute_force_on_long_realizations() {
    let mut rng = Lcg::new(0xDEAD_BEEF_1234_5678);
    for (d, count, decay) in [(2usize, 400usize, 0.9), (3, 300, 0.05), (5, 200, 3.0)] {
        assert!(count * d <= MAX_EVENTS_FOR_COMPARISON);
        let horizon = count as f64 / 2.0;
        let events: Vec<Vec<f64>> = (0..d)
            .map(|_| {
                let mut times: Vec<f64> = (0..count).map(|_| rng.next_f64() * horizon).collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times
            })
            .collect();
        let baseline: Vec<f64> = (0..d).map(|i| 0.3 + 0.1 * i as f64).collect();
        let excitation: Vec<f64> = (0..d * d)
            .map(|k| 0.05 + 0.4 * (k % 3) as f64 / d as f64)
            .collect();
        let p = parameters(baseline, excitation, decay);
        let observation = Observation::new(&events, horizon).unwrap();
        assert_agrees(&p, &observation, &format!("d={d} n={count} beta={decay}"));
    }
}

// ------------------------------------------------------------ gate sensitivity
//
// Replicates `gate_sensitivity.rs` for the multivariate scale. Same reasoning: the
// denominator is guarded in the direction that does not announce itself, because
// inflating it loosens the gate and a looser gate does not fail.

/// `multivariate_computation_scale` for `mu = [0.3, 0.3]`,
/// `alpha = [[0.7, 0.7], [0.35, 0.7]]`, `beta = 4`, `T = 10`,
/// `events = [[0, 0.05, 0.1, 3, 6], [0.02, 0.07, 4, 8]]`.
///
/// Hand calculation: `sum_i mu_i*T = 6`, the paired compensator term is
/// `10.849530234105423`, and `sum |ln lambda| = 11.816066600877548`.
const HAND_SCALE_MIXED_SIGNS: f64 = 28.66559683498297;

/// `nll` for the same case. The ratio matters: `|nll| / scale = 0.5895`, so replacing
/// the denominator with `|nll|` moves the gate's boundary to 59% of where it belongs.
const HAND_NLL_MIXED_SIGNS: f64 = 16.898952862005846;

/// The same case with `|sum ln lambda|` in place of `sum |ln lambda|`. The signed sum
/// is `-0.04942262790042168`, a factor of 239 smaller than the magnitudes.
const SIGNED_SUM_VARIANT: f64 = 16.899024862005846;

fn mixed_sign_case() -> (Parameters, Vec<Vec<f64>>, f64) {
    (
        parameters(vec![0.3, 0.3], vec![0.7, 0.7, 0.35, 0.7], 4.0),
        vec![vec![0.0, 0.05, 0.1, 3.0, 6.0], vec![0.02, 0.07, 4.0, 8.0]],
        10.0,
    )
}

#[test]
fn multivariate_scale_matches_a_hand_calculation() {
    let (p, events, horizon) = mixed_sign_case();
    let observation = Observation::new(&events, horizon).unwrap();
    let scale = scale_of(&p, &observation);
    assert!(
        (scale - HAND_SCALE_MIXED_SIGNS).abs() <= 1e-12 * HAND_SCALE_MIXED_SIGNS,
        "multivariate_computation_scale gave {scale:?}, hand calculation gives \
         {HAND_SCALE_MIXED_SIGNS:?}. Taking |sum ln lambda| instead of \
         sum |ln lambda| would give about {SIGNED_SUM_VARIANT:?}."
    );
    let nll = negative_log_likelihood(&p, &observation);
    assert!(
        (nll - HAND_NLL_MIXED_SIGNS).abs() <= 1e-12 * HAND_NLL_MIXED_SIGNS,
        "nll gave {nll:?}, hand calculation gives {HAND_NLL_MIXED_SIGNS:?}"
    );
}

/// The gate's sensitivity, pinned from both sides against the hand-calculated scale
/// rather than against whatever `scale_of` currently returns.
#[test]
fn the_multivariate_gate_rejects_just_above_its_sensitivity() {
    let (p, events, horizon) = mixed_sign_case();
    let observation = Observation::new(&events, horizon).unwrap();
    let recursive = negative_log_likelihood(&p, &observation);
    let reference = brute_force(&p, &observation);
    let scale = scale_of(&p, &observation);
    let boundary = RECURSION_TOLERANCE * HAND_SCALE_MIXED_SIGNS;

    let gate_accepts = |value: f64| (value - reference).abs() <= RECURSION_TOLERANCE * scale;

    assert!(
        gate_accepts(recursive + 0.9 * boundary),
        "the gate rejected an error below its documented sensitivity; the \
         denominator has SHRUNK. Replacing it with |nll| ({HAND_NLL_MIXED_SIGNS:?}) \
         would put the boundary at {:e} instead of {boundary:e}.",
        RECURSION_TOLERANCE * HAND_NLL_MIXED_SIGNS
    );
    assert!(
        !gate_accepts(recursive + 1.1 * boundary),
        "the gate ACCEPTED an error above its documented sensitivity of \
         {boundary:e}; the denominator has GROWN and the comparison is weaker than \
         it claims"
    );
    assert!(gate_accepts(recursive - 0.9 * boundary));
    assert!(!gate_accepts(recursive - 1.1 * boundary));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn agrees_with_brute_force_over_random_parameters(
        d in 1usize..=5,
        seed in 0u64..100_000,
        horizon in 2.0f64..40.0,
        tie_grid in prop::bool::ANY,
    ) {
        let mut rng = Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let events: Vec<Vec<f64>> = (0..d)
            .map(|_| {
                let n = rng.next_usize(30);
                let mut times: Vec<f64> = (0..n)
                    .map(|_| if tie_grid {
                        (rng.next_f64() * 6.0).floor() * horizon / 6.0
                    } else {
                        rng.next_f64() * horizon
                    })
                    .collect();
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                times
            })
            .collect();
        let baseline: Vec<f64> = (0..d).map(|_| 0.05 + rng.next_f64() * 2.0).collect();
        let excitation: Vec<f64> =
            (0..d * d).map(|_| rng.next_f64() * 0.8 / d as f64).collect();
        let decay = 0.05 + rng.next_f64() * 4.0;

        let p = Parameters::new(baseline, excitation, decay).unwrap();
        let observation = Observation::new(&events, horizon).unwrap();

        let recursive = negative_log_likelihood(&p, &observation);
        let reference = brute_force(&p, &observation);
        let scale = scale_of(&p, &observation);
        let discrepancy = (recursive - reference).abs();
        prop_assert!(
            discrepancy <= RECURSION_TOLERANCE * scale,
            "d={} recursion {:?} vs definition {:?}, |difference| {:e} > {:e}",
            d, recursive, reference, discrepancy, RECURSION_TOLERANCE * scale
        );
    }
}

/// A `Parameters` and an `Observation` that describe different processes must not be
/// evaluated together.
///
/// Before this check existed, `d = 3` parameters with two components of events
/// **silently returned a number** — the missing component was treated as empty — and
/// the other direction read past the end of the counts array and panicked with `index
/// out of bounds`. Both were found by the Python bindings' error-mapping test, which
/// requires that no panic reach the interpreter.
///
/// The Rust side treats this as a documented invariant of the pair and panics
/// (CLAUDE.md §5); `hawk-python` checks first and raises `ValueError`.
#[test]
#[should_panic(expected = "they must agree")]
fn too_few_event_components_is_not_silently_accepted() {
    let p = parameters(vec![0.4, 0.4, 0.4], vec![0.1; 9], 1.0);
    let events = vec![vec![1.0], vec![2.0]];
    let observation = Observation::new(&events, 5.0).unwrap();
    negative_log_likelihood(&p, &observation);
}

#[test]
#[should_panic(expected = "they must agree")]
fn too_many_event_components_is_not_an_out_of_bounds_read() {
    let p = parameters(vec![0.4, 0.4], vec![0.1; 4], 1.0);
    let events = vec![vec![1.0], vec![2.0], vec![3.0]];
    let observation = Observation::new(&events, 5.0).unwrap();
    negative_log_likelihood(&p, &observation);
}
