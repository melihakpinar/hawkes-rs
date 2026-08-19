//! Meta-tests for the step 7 comparison gate's **denominator** (issue #7, gap a).
//!
//! `computation_scale` is the denominator of the recursion-versus-definition
//! comparison in `loglikelihood.rs`. Until now nothing guarded it, and it is guarded
//! in the direction that matters: **inflating it silently loosens the gate**, and a
//! looser gate does not fail, it just stops catching things. That is the one class of
//! regression a test suite cannot notice by running.
//!
//! Two failure modes are anticipated, and both change the gate's *sensitivity*:
//!
//! 1. Someone multiplies the denominator, or adds a term to it, and the gate quietly
//!    admits errors it used to reject.
//! 2. Someone "simplifies" it back to `|nll|`, which reads as the obvious thing to
//!    write and is wrong wherever the terms cancel.
//!
//! The tests below pin the sensitivity to numbers derived **independently of
//! `computation_scale`** — hand calculations, worked out in a separate
//! implementation and written out here. Pinning it to a value the function itself
//! produced would move with the bug and see nothing.
//!
//! # Sabotage
//!
//! Multiplying the denominator by 10 turned
//! `the_gate_rejects_just_above_its_sensitivity` red, and
//! `computation_scale_matches_a_hand_calculation` with it. Replacing the sum of
//! magnitudes with the magnitude of the sum turned
//! `the_denominator_sums_magnitudes_not_the_signed_sum` red. Recorded in
//! `docs/verification-log.md`.

mod common;

use common::{
    RECURSION_TOLERANCE, brute_force_negative_log_likelihood as brute_force, computation_scale,
};
use hawk::univariate::{Observation, Parameters, negative_log_likelihood};

/// `computation_scale` for `mu = 2, alpha = 0.5, beta = 1.5, T = 3, t = [1, 2]`.
///
/// Hand calculation, from the same working as `reference_loglikelihood.rs`'s
/// `matches_hand_calculation_two_events`:
///
/// ```text
///   mu*T                      = 6
///   compensator_excitation    = (1 - e^-3) + (1 - e^-1.5)
///                             = 0.950212931632136 + 0.7768698398515702
///                             = 1.7270827714837063
///   alpha * compensator       = 0.8635413857418531
///   lambda(1) = 2,  lambda(2) = 2.167347620111322
///   |ln 2| + |ln 2.167347620111322|
///                             = 0.6931471805599453 + 0.7735041250611435
///                             = 1.4666513056210888
///   scale = 6 + 0.8635413857418531 + 1.4666513056210888
/// ```
const HAND_SCALE_TWO_EVENTS: f64 = 8.330192691362942;

/// `nll` for the same case, from `reference_loglikelihood.rs`.
///
/// Recorded here because the ratio is the point: `|nll| / scale = 0.6479`. If the
/// denominator were replaced by `|nll|` the gate's boundary would move to 65% of
/// where it belongs, which the bracket below is tight enough to catch.
const HAND_NLL_TWO_EVENTS: f64 = 5.396890080120764;

fn two_event_case() -> (Parameters, [f64; 2], f64) {
    (Parameters::new(2.0, 0.5, 1.5).unwrap(), [1.0, 2.0], 3.0)
}

#[test]
fn computation_scale_matches_a_hand_calculation() {
    let (parameters, times, horizon) = two_event_case();
    let observation = Observation::new(&times, horizon).unwrap();
    let scale = computation_scale(&parameters, &observation);
    let discrepancy = (scale - HAND_SCALE_TWO_EVENTS).abs();
    assert!(
        discrepancy <= 1e-12 * HAND_SCALE_TWO_EVENTS,
        "computation_scale gave {scale:?}, hand calculation gives \
         {HAND_SCALE_TWO_EVENTS:?}. The denominator of the step 7 gate has changed; \
         if that was deliberate, the sensitivity tests in this file need redoing."
    );
}

/// The gate's sensitivity, pinned from both sides.
///
/// The comparison in `loglikelihood.rs` is
/// `|recursive - brute| <= RECURSION_TOLERANCE * scale`, so injecting an absolute
/// error `delta` must be accepted below `RECURSION_TOLERANCE * scale` and rejected
/// above it. The boundary is computed from [`HAND_SCALE_TWO_EVENTS`], not from
/// `computation_scale`, which is what makes this able to see the denominator move.
///
/// The bracket is deliberately tight — 0.9x and 1.1x. It has to be: replacing the
/// denominator with `|nll|` moves the boundary to 0.6479x, and a bracket of 0.5x/2x
/// would sit right across that and could miss it. The intrinsic disagreement between
/// the two implementations on this case is around `1e-16` absolute, four orders below
/// the boundary at `8.3e-12`, so there is no flakiness risk in tightening it.
#[test]
fn the_gate_rejects_just_above_its_sensitivity() {
    let (parameters, times, horizon) = two_event_case();
    let observation = Observation::new(&times, horizon).unwrap();

    let recursive = negative_log_likelihood(&parameters, &observation);
    let reference = brute_force(&parameters, &observation);
    let scale = computation_scale(&parameters, &observation);
    let boundary = RECURSION_TOLERANCE * HAND_SCALE_TWO_EVENTS;

    // Exactly the predicate `loglikelihood.rs` applies.
    let gate_accepts = |value: f64| (value - reference).abs() <= RECURSION_TOLERANCE * scale;

    assert!(
        gate_accepts(recursive + 0.9 * boundary),
        "the gate rejected an error of {:e}, which is below its documented \
         sensitivity of {boundary:e}. The denominator has SHRUNK -- if it was \
         replaced by |nll| ({HAND_NLL_TWO_EVENTS:?}) the boundary would sit at \
         {:e}, which is exactly this symptom.",
        0.9 * boundary,
        RECURSION_TOLERANCE * HAND_NLL_TWO_EVENTS,
    );

    assert!(
        !gate_accepts(recursive + 1.1 * boundary),
        "the gate ACCEPTED an error of {:e}, above its documented sensitivity of \
         {boundary:e}. The denominator has GROWN, so the step 7 comparison is now \
         weaker than it claims to be. This is the failure mode that does not \
         announce itself: a loosened gate does not fail, it stops catching things.",
        1.1 * boundary,
    );

    // Both signs, so a denominator error cannot hide behind the direction of the
    // injected perturbation.
    assert!(gate_accepts(recursive - 0.9 * boundary));
    assert!(!gate_accepts(recursive - 1.1 * boundary));
}

/// `computation_scale` for `mu = 0.3, alpha = 0.9, beta = 8, T = 10` on
/// `t = [0, 0.05, 0.1, 0.15, 2, 4, 6, 8]`.
///
/// Hand calculation:
///
/// ```text
///   mu*T                   = 3.0
///   alpha * compensator    = 7.199999898718332
///   sum_k |ln lambda_k|    = 12.132088887915735
///   scale                  = 22.332088786634067
/// ```
const HAND_SCALE_MIXED_SIGNS: f64 = 22.332088786634067;

/// The same case with `|sum_k ln lambda_k|` in place of `sum_k |ln lambda_k|`.
///
/// ```text
///   sum_k ln lambda_k      = 0.0924204652106635      <- 131x smaller
///   3.0 + 7.199999898718332 + 0.0924204652106635
/// ```
const SIGNED_SUM_VARIANT: f64 = 10.292420363928996;

/// The third term must sum **magnitudes**, not take the magnitude of the sum.
///
/// This is the same cancellation defect as the outer one, one level down: `ln lambda`
/// is negative wherever `lambda < 1`, which is routine for `mu < 1`, so a near-zero
/// `log_term` can be the cancellation of many `O(1)` contributions. The case below is
/// built to make that visible: the signed sum is `0.0924` while the magnitudes total
/// `12.13`, a factor of 131.
#[test]
fn the_denominator_sums_magnitudes_not_the_signed_sum() {
    let parameters = Parameters::new(0.3, 0.9, 8.0).unwrap();
    let times = [0.0, 0.05, 0.1, 0.15, 2.0, 4.0, 6.0, 8.0];
    let observation = Observation::new(&times, 10.0).unwrap();

    let scale = computation_scale(&parameters, &observation);
    let discrepancy = (scale - HAND_SCALE_MIXED_SIGNS).abs();
    assert!(
        discrepancy <= 1e-12 * HAND_SCALE_MIXED_SIGNS,
        "computation_scale gave {scale:?}, hand calculation gives \
         {HAND_SCALE_MIXED_SIGNS:?}. Taking |sum ln lambda| instead of \
         sum |ln lambda| would give {SIGNED_SUM_VARIANT:?} here -- less than half, \
         because the log terms very nearly cancel."
    );

    // Guards the guard: if the case ever stopped having mixed-sign log terms, the
    // test above would still pass and would no longer be testing anything.
    assert!(
        scale > 2.0 * SIGNED_SUM_VARIANT,
        "this case no longer discriminates between the two forms; pick another"
    );
}
