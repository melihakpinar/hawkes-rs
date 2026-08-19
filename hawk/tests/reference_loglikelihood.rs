//! Validates the brute-force reference itself (M1 Part B step 5).
//!
//! The reference is what every other likelihood test is measured against, so it
//! cannot be checked against `hawk` — that would be circular. Every expected value
//! below is either a hand calculation written out in the test, or an identity that
//! holds independently of this crate.
//!
//! # Sabotage
//!
//! Changing the inner comparison from `t_i < t_k` to `t_i <= t_k` turned
//! `matches_hand_calculation_two_events` red (`5.274115` -> `4.856...`), as did
//! dropping the `beta` factor from the kernel. Recorded in
//! `docs/verification-log.md`.

mod common;

use common::brute_force_negative_log_likelihood as brute_force;
use hawk::univariate::{Observation, Parameters};

/// Agreement required against a hand calculation.
///
/// The expected values below are decimal literals transcribed from an independent
/// evaluation, so they carry the `f64` rounding of that evaluation. 1e-12 relative is
/// far inside what a real transcription error would produce — the sabotage cases move
/// the fourth significant figure — while leaving room for the last-bit differences
/// between two orderings of the same arithmetic.
const HAND_CALCULATION_TOLERANCE: f64 = 1e-12;

fn close(actual: f64, expected: f64, tolerance: f64) -> bool {
    (actual - expected).abs() <= tolerance * f64::max(1.0, expected.abs())
}

#[test]
fn matches_hand_calculation_two_events() {
    // mu = 2, alpha = 0.5, beta = 1.5, T = 3, events = [1, 2].
    //
    //   compensator = mu*T + alpha*[ (1 - e^-3) + (1 - e^-1.5) ]
    //               = 6 + 0.5*[ 0.950212931632136 + 0.7768698398515702 ]
    //               = 6.863541385741853
    //   lambda(1)   = mu = 2                    (predictable: nothing earlier)
    //   lambda(2)   = 2 + 0.5*1.5*e^-1.5 = 2.167347620111322
    //   log_term    = ln(2) + ln(2.167347620111322) = 1.4666513056210888
    //   nll         = 6.863541385741853 - 1.4666513056210888
    //
    // beta is deliberately not 1. An earlier version of this test used beta = 1,
    // where `alpha*beta` and `alpha` coincide, so it could not see a kernel missing
    // its beta factor -- sabotage S11 passed against it. With beta = 1.5 the same
    // sabotage gives 5.422964788539137 and the test fails.
    let parameters = Parameters::new(2.0, 0.5, 1.5).unwrap();
    let times = [1.0, 2.0];
    let observation = Observation::new(&times, 3.0).unwrap();

    let actual = brute_force(&parameters, &observation);
    let expected = 5.396890080120764;
    assert!(
        close(actual, expected, HAND_CALCULATION_TOLERANCE),
        "brute force gave {actual:?}, hand calculation gives {expected:?}"
    );
}

#[test]
fn matches_hand_calculation_one_event() {
    // mu = 1.5, alpha = 0.4, beta = 2, T = 4, events = [1].
    // With one event the intensity at it is exactly mu, so
    //   nll = 1.5*4 + 0.4*(1 - e^-6) - ln(1.5)
    let parameters = Parameters::new(1.5, 0.4, 2.0).unwrap();
    let times = [1.0];
    let observation = Observation::new(&times, 4.0).unwrap();

    let actual = brute_force(&parameters, &observation);
    let expected = 5.993543391021169;
    assert!(
        close(actual, expected, HAND_CALCULATION_TOLERANCE),
        "brute force gave {actual:?}, hand calculation gives {expected:?}"
    );
}

#[test]
fn empty_realization_is_the_compensator_alone() {
    // No events: no log term, and no kernel contributes. nll = mu*T exactly.
    let parameters = Parameters::new(0.7, 0.3, 1.1).unwrap();
    let observation = Observation::new(&[], 5.0).unwrap();

    let actual = brute_force(&parameters, &observation);
    assert!(
        close(actual, 3.5, HAND_CALCULATION_TOLERANCE),
        "brute force gave {actual:?}, expected mu*T = 3.5"
    );
}

/// As `alpha -> 0` the process degenerates to a homogeneous Poisson process, whose
/// negative log-likelihood has the closed form `mu*T - n*ln(mu)`.
///
/// This identity comes from the Poisson likelihood, not from this crate, so it checks
/// the whole expression — compensator, log term, and the kernel's disappearance —
/// against something external. It is the same degenerate case that pinned `tick`'s
/// loss convention in M0 (`conventions.md` C7).
#[test]
fn degenerates_to_the_poisson_likelihood() {
    let mu = 1.7;
    let horizon = 9.0;
    let times = [0.5, 1.25, 3.0, 3.0, 7.75, 8.5];
    let observation = Observation::new(&times, horizon).unwrap();
    let poisson = mu * horizon - (times.len() as f64) * mu.ln();

    // alpha cannot be zero (Parameters requires positivity), so approach it. The
    // remaining discrepancy is O(alpha) by construction: every excitation term is
    // proportional to alpha.
    for excitation in [1e-8, 1e-10, 1e-12] {
        let parameters = Parameters::new(mu, excitation, 1.3).unwrap();
        let actual = brute_force(&parameters, &observation);
        let discrepancy = (actual - poisson).abs();
        assert!(
            discrepancy < 20.0 * excitation,
            "alpha={excitation:e}: brute force {actual:?} vs Poisson {poisson:?}, \
             discrepancy {discrepancy:e} is not O(alpha)"
        );
    }
}

/// Tied events must not excite one another (`conventions.md` C3, C8).
///
/// The expected value is the one worked out in
/// `univariate_loglikelihood.md` §8, where the textbook recursion gives
/// `5.406576697862245` instead — a 9% error. This test pins the correct number
/// independently of any recursion.
#[test]
fn tied_events_do_not_excite_each_other() {
    let parameters = Parameters::new(0.7, 0.5, 1.3).unwrap();
    let times = [1.0, 2.0, 2.0, 3.0];
    let observation = Observation::new(&times, 5.0).unwrap();

    let actual = brute_force(&parameters, &observation);
    let expected = 5.961059318008664;
    assert!(
        close(actual, expected, HAND_CALCULATION_TOLERANCE),
        "brute force gave {actual:?}, expected {expected:?}. The textbook recursion \
         would give 5.406576697862245 here."
    );
}

#[test]
fn rejects_invalid_input() {
    assert!(
        Parameters::new(0.0, 0.5, 1.0).is_err(),
        "mu = 0 must be rejected"
    );
    assert!(
        Parameters::new(1.0, -0.5, 1.0).is_err(),
        "alpha < 0 must be rejected"
    );
    assert!(
        Parameters::new(1.0, 0.5, f64::NAN).is_err(),
        "NaN must be rejected"
    );

    // Unsorted input is an error, not something to silently sort (C8). `tick`
    // accepts it and returns a different wrong number per ordering.
    assert!(
        Observation::new(&[2.0, 1.0], 5.0).is_err(),
        "descending timestamps must be rejected"
    );
    assert!(
        Observation::new(&[1.0, 6.0], 5.0).is_err(),
        "a timestamp beyond the horizon must be rejected"
    );
    assert!(
        Observation::new(&[-1.0, 1.0], 5.0).is_err(),
        "a negative timestamp must be rejected"
    );
    assert!(
        Observation::new(&[1.0], 0.0).is_err(),
        "horizon 0 must be rejected"
    );

    // Ties and the window endpoints are accepted.
    assert!(Observation::new(&[0.0, 2.0, 2.0, 5.0], 5.0).is_ok());
}
