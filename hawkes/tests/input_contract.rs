//! The input contract and the error paths of the public API (issue #46).
//!
//! `conventions.md` C8 fixes the contract — ascending within a component, every
//! timestamp in `[0, T]` endpoints included, ties permitted within and across
//! components, a component may be empty — and C5 makes `T` the caller's to supply.
//! `hawkes/src/error.rs` documents which variant each violation produces. Until this
//! file most of those variants had no Rust test: nothing exercised `NonFiniteEvent`,
//! any `multivariate::Observation::new` or `multivariate::Parameters::new` error,
//! `simulate` rejecting a bad horizon, or `fit` / `fit_from` rejecting too few events
//! or a wrong start length.
//!
//! # Every single case has a randomised counterpart
//!
//! CLAUDE.md §3: a fixed case exercises one input. Each sweep here draws valid input,
//! injects **exactly one** violation at a random position, and requires the documented
//! variant with the documented fields. One violation at a time on purpose: the order in
//! which the validators run is not part of the contract, and a case that violated two
//! rules at once would pin it.
//!
//! # Expected values
//!
//! Error variants and their fields are the contract in `error.rs`. The few numeric
//! expectations — an event at exactly the horizon, at exactly zero, and the compensator
//! on tied input — are hand calculations from the definitions, recorded in
//! `docs/derivations/edge_case_hand_calculations.md`. Nothing here was computed with
//! the code under test.
//!
//! # Sabotage
//!
//! Each validator was disabled in turn — 31 mutations, listed in
//! `docs/verification-log.md` S53 — and the tests that name it went red, the fixed case
//! and its sweep together, while the rest of the file stayed green. Two results worth
//! keeping here: disabling only the `horizon <= 0.0` half of the horizon check turned
//! `univariate_observation_rejects_a_bad_horizon` red and nothing else, including the
//! older `rejects_invalid_input`, whose `horizon = 0` case had been passing for the wrong
//! reason (its one event was already past a zero horizon); and removing the data bound
//! from `multivariate::fit` alone turned nothing red, because `fit_from` re-checks it.

mod common;

use hawkes::Error;
use hawkes::multivariate;
use hawkes::univariate;
use proptest::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Agreement required against the hand calculations in
/// `edge_case_hand_calculations.md`. Same bound and reasoning as
/// `reference_loglikelihood.rs`: the expected values carry one evaluation's `f64`
/// rounding, and a transcription error moves the fourth significant figure.
const HAND_CALCULATION_TOLERANCE: f64 = 1e-12;

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= HAND_CALCULATION_TOLERANCE * f64::max(1.0, expected.abs())
}

/// The smallest `f64` strictly above a positive `value`. Written out because
/// `f64::next_up` is newer than the crate's MSRV.
fn one_ulp_above(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

/// Sorted timestamps in `[0, horizon]`. On the grid, every value is a multiple of
/// `horizon / 8`, so ties, exact zeros and events at exactly the horizon are common
/// rather than incidental.
fn sorted_times(raw: &[f64], horizon: f64, on_grid: bool) -> Vec<f64> {
    let mut times: Vec<f64> = raw
        .iter()
        .map(|u| {
            if on_grid {
                (u * 8.0).round() / 8.0 * horizon
            } else {
                u * horizon
            }
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times
}

// ------------------------------------------------------------ univariate::Parameters

#[test]
fn univariate_parameters_reject_non_positive_values() {
    for (baseline, excitation, decay, name, value) in [
        (0.0, 0.5, 1.0, "baseline", 0.0),
        (-1.0, 0.5, 1.0, "baseline", -1.0),
        (1.0, 0.0, 1.0, "excitation", 0.0),
        (1.0, -0.5, 1.0, "excitation", -0.5),
        (1.0, 0.5, 0.0, "decay", 0.0),
        (1.0, 0.5, -2.0, "decay", -2.0),
    ] {
        assert_eq!(
            univariate::Parameters::new(baseline, excitation, decay),
            Err(Error::NonPositiveParameter { name, value }),
            "({baseline}, {excitation}, {decay})"
        );
    }
}

#[test]
fn univariate_parameters_reject_non_finite_values() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (slot, name) in [(0, "baseline"), (1, "excitation"), (2, "decay")] {
            let mut values = [1.0, 0.5, 1.0];
            values[slot] = bad;
            let error = univariate::Parameters::new(values[0], values[1], values[2])
                .expect_err("a non-finite parameter must be rejected");
            assert!(
                matches!(
                    error,
                    Error::NonFiniteParameter { name: got, value }
                        if got == name && (value.is_nan() == bad.is_nan()) && (value.is_nan() || value == bad)
                ),
                "{name} = {bad}: got {error:?}"
            );
        }
    }
}

#[test]
fn univariate_parameters_accept_positive_finite_values() {
    let parameters = univariate::Parameters::new(0.5, 0.6, 1.0).unwrap();
    assert_eq!(parameters.baseline().to_bits(), 0.5f64.to_bits());
    assert_eq!(parameters.excitation().to_bits(), 0.6f64.to_bits());
    assert_eq!(parameters.decay().to_bits(), 1.0f64.to_bits());
}

// ----------------------------------------------------------- univariate::Observation

#[test]
fn univariate_observation_accepts_the_window_endpoints_and_ties() {
    let times = [0.0, 2.0, 2.0, 5.0];
    let observation = univariate::Observation::new(&times, 5.0).unwrap();
    assert_eq!(observation.len(), 4);
    assert!(!observation.is_empty());
    assert_eq!(observation.times(), &times);
    assert_eq!(observation.horizon().to_bits(), 5.0f64.to_bits());
}

#[test]
fn univariate_observation_with_no_events_is_valid() {
    let observation = univariate::Observation::new(&[], 5.0).unwrap();
    assert_eq!(observation.len(), 0);
    assert!(observation.is_empty());
}

#[test]
fn univariate_observation_rejects_an_event_one_ulp_past_the_horizon() {
    let horizon = 5.0;
    let past = one_ulp_above(horizon);
    assert_eq!(
        univariate::Observation::new(&[1.0, past], horizon).map(|_| ()),
        Err(Error::EventOutsideWindow {
            index: 1,
            time: past,
            horizon,
        })
    );
    // The contract is "shorter horizon than the last event", not "far shorter".
    assert_eq!(
        univariate::Observation::new(&[1.0, 6.0], horizon).map(|_| ()),
        Err(Error::EventOutsideWindow {
            index: 1,
            time: 6.0,
            horizon,
        })
    );
    assert_eq!(
        univariate::Observation::new(&[-1e-300, 1.0], horizon).map(|_| ()),
        Err(Error::EventOutsideWindow {
            index: 0,
            time: -1e-300,
            horizon,
        })
    );
}

#[test]
fn univariate_observation_rejects_unsorted_input() {
    // Rejected, never sorted on the caller's behalf (C8; `error.rs`).
    assert_eq!(
        univariate::Observation::new(&[2.0, 1.0], 5.0).map(|_| ()),
        Err(Error::UnsortedEvents {
            index: 1,
            previous_index: 0,
            previous: 2.0,
            current: 1.0,
        })
    );
    assert_eq!(
        univariate::Observation::new(&[1.0, 3.0, 2.0, 4.0], 5.0).map(|_| ()),
        Err(Error::UnsortedEvents {
            index: 2,
            previous_index: 1,
            previous: 3.0,
            current: 2.0,
        })
    );
}

#[test]
fn univariate_observation_rejects_non_finite_events() {
    assert_eq!(
        univariate::Observation::new(&[1.0, f64::NAN, 3.0], 5.0).map(|_| ()),
        Err(Error::NonFiniteEvent { index: 1 })
    );
    // An infinite timestamp is both non-finite and outside the window. Both
    // descriptions in `error.rs` apply and the contract does not order them, so either
    // variant is accepted as long as it names the right index.
    for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
        let error = univariate::Observation::new(&[infinite, 3.0], 5.0)
            .expect_err("an infinite timestamp must be rejected");
        assert!(
            matches!(
                error,
                Error::NonFiniteEvent { index: 0 } | Error::EventOutsideWindow { index: 0, .. }
            ),
            "{infinite}: got {error:?}"
        );
    }
}

#[test]
fn univariate_observation_rejects_a_bad_horizon() {
    for horizon in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
        for times in [&[][..], &[1.0][..]] {
            assert_eq!(
                univariate::Observation::new(times, horizon).map(|_| ()),
                Err(Error::InvalidHorizon { horizon }),
                "horizon {horizon} with {times:?}"
            );
        }
    }
    assert!(matches!(
        univariate::Observation::new(&[], f64::NAN),
        Err(Error::InvalidHorizon { horizon }) if horizon.is_nan()
    ));
}

// ------------------------------------------------ univariate values at the window edge

/// `edge_case_hand_calculations.md` A.
#[test]
fn an_event_at_exactly_the_horizon_matches_the_hand_calculation() {
    let parameters = univariate::Parameters::new(1.5, 0.4, 2.0).unwrap();
    let observation = univariate::Observation::new(&[4.0], 4.0).unwrap();

    let value = univariate::negative_log_likelihood(&parameters, &observation);
    assert!(
        close(value, 5.594534891891835),
        "nll {value:?}, hand calculation 5.594534891891835"
    );

    let (again, gradient) =
        univariate::negative_log_likelihood_and_gradient(&parameters, &observation);
    assert_eq!(value.to_bits(), again.to_bits());
    assert!(close(gradient.baseline, 3.3333333333333335), "{gradient:?}");
    // `alpha` and `beta` do not appear in `mu*T - ln(mu)`, so these are exactly zero.
    assert!(
        gradient.excitation.abs() <= HAND_CALCULATION_TOLERANCE,
        "{gradient:?}"
    );
    assert!(
        gradient.decay.abs() <= HAND_CALCULATION_TOLERANCE,
        "{gradient:?}"
    );
}

/// `edge_case_hand_calculations.md` B.
#[test]
fn an_event_at_exactly_zero_matches_the_hand_calculation() {
    let parameters = univariate::Parameters::new(1.5, 0.4, 2.0).unwrap();
    let observation = univariate::Observation::new(&[0.0], 4.0).unwrap();

    let (value, gradient) =
        univariate::negative_log_likelihood_and_gradient(&parameters, &observation);
    assert!(close(value, 5.994400706840675), "nll {value:?}");
    assert!(close(gradient.baseline, 3.3333333333333335), "{gradient:?}");
    assert!(
        close(gradient.excitation, 0.9996645373720975),
        "{gradient:?}"
    );
    assert!(close(gradient.decay, 0.000536740204644019), "{gradient:?}");
}

/// `edge_case_hand_calculations.md` D: tied events do not contribute to each other's
/// compensator value.
#[test]
fn the_compensator_at_tied_events_matches_the_hand_calculation() {
    let parameters = univariate::Parameters::new(0.7, 0.5, 1.3).unwrap();
    let observation = univariate::Observation::new(&[1.0, 2.0, 2.0, 3.0], 5.0).unwrap();
    let actual = univariate::compensator_at_events(&parameters, &observation);
    let expected = [
        0.7,
        1.7637341034829936,
        1.7637341034829936,
        3.29033141785882,
    ];
    assert_eq!(actual.len(), expected.len());
    for (k, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            close(a, e),
            "Lambda(t_{k}): got {a:?}, hand calculation {e:?}"
        );
    }
    assert!(
        univariate::compensator_at_events(
            &parameters,
            &univariate::Observation::new(&[], 5.0).unwrap()
        )
        .is_empty()
    );
}

// --------------------------------------------------------- multivariate::Parameters

#[test]
fn multivariate_parameters_reject_an_empty_process() {
    assert_eq!(
        multivariate::Parameters::new(vec![], vec![], 1.0),
        Err(Error::EmptyProcess)
    );
}

#[test]
fn multivariate_parameters_reject_a_non_square_excitation() {
    assert_eq!(
        multivariate::Parameters::new(vec![0.5, 0.5], vec![0.1, 0.1, 0.1], 1.0),
        Err(Error::DimensionMismatch {
            what: "excitation",
            actual: 3,
            expected: 4,
            dimension: 2,
        })
    );
}

#[test]
fn multivariate_parameters_reject_a_bad_baseline_entry() {
    assert_eq!(
        multivariate::Parameters::new(vec![0.5, 0.0], vec![0.1; 4], 1.0),
        Err(Error::NonPositiveParameter {
            name: "baseline",
            value: 0.0,
        })
    );
    assert_eq!(
        multivariate::Parameters::new(vec![0.5, -0.25], vec![0.1; 4], 1.0),
        Err(Error::NonPositiveParameter {
            name: "baseline",
            value: -0.25,
        })
    );
    assert!(matches!(
        multivariate::Parameters::new(vec![0.5, f64::NAN], vec![0.1; 4], 1.0),
        Err(Error::NonFiniteParameter { name: "baseline", value }) if value.is_nan()
    ));
}

/// Zero entries are legitimate in `d` dimensions; negative and non-finite ones are
/// not. The row and column in the error follow the row-major layout of C6.
#[test]
fn multivariate_parameters_reject_a_bad_excitation_entry() {
    assert!(multivariate::Parameters::new(vec![0.5; 3], vec![0.0; 9], 1.0).is_ok());
    let mut excitation = vec![0.1; 9];
    excitation[5] = -0.1; // row 1, column 2
    assert_eq!(
        multivariate::Parameters::new(vec![0.5; 3], excitation, 1.0),
        Err(Error::InvalidExcitation {
            row: 1,
            column: 2,
            value: -0.1,
        })
    );
    let mut excitation = vec![0.1; 9];
    excitation[7] = f64::INFINITY; // row 2, column 1
    assert_eq!(
        multivariate::Parameters::new(vec![0.5; 3], excitation, 1.0),
        Err(Error::InvalidExcitation {
            row: 2,
            column: 1,
            value: f64::INFINITY,
        })
    );
}

#[test]
fn multivariate_parameters_reject_a_bad_decay() {
    assert_eq!(
        multivariate::Parameters::new(vec![0.5], vec![0.1], 0.0),
        Err(Error::NonPositiveParameter {
            name: "decay",
            value: 0.0,
        })
    );
    assert_eq!(
        multivariate::Parameters::new(vec![0.5], vec![0.1], f64::NEG_INFINITY),
        Err(Error::NonFiniteParameter {
            name: "decay",
            value: f64::NEG_INFINITY,
        })
    );
}

// -------------------------------------------------------- multivariate::Observation

#[test]
fn multivariate_observation_accepts_endpoints_ties_and_empty_components() {
    let events = vec![vec![0.0, 2.5, 6.0], vec![2.5, 2.5], vec![]];
    let observation = multivariate::Observation::new(&events, 6.0).unwrap();
    assert_eq!(observation.dimension(), 3);
    assert_eq!(observation.len(), 5);
    assert!(!observation.is_empty());
    assert_eq!(observation.events(), &events[..]);

    let all_empty = vec![vec![], vec![]];
    let observation = multivariate::Observation::new(&all_empty, 6.0).unwrap();
    assert_eq!(observation.len(), 0);
    assert!(observation.is_empty());
}

#[test]
fn multivariate_observation_rejects_an_empty_component_list() {
    assert_eq!(
        multivariate::Observation::new(&[], 5.0).map(|_| ()),
        Err(Error::EmptyProcess)
    );
}

#[test]
fn multivariate_observation_rejects_a_bad_horizon() {
    let events = vec![vec![1.0], vec![]];
    for horizon in [0.0, -3.0, f64::INFINITY] {
        assert_eq!(
            multivariate::Observation::new(&events, horizon).map(|_| ()),
            Err(Error::InvalidHorizon { horizon })
        );
    }
    assert!(matches!(
        multivariate::Observation::new(&events, f64::NAN),
        Err(Error::InvalidHorizon { horizon }) if horizon.is_nan()
    ));
}

/// The index in each error is the position **within the offending component**; the
/// contract is per component (C8) and so is the report.
#[test]
fn multivariate_observation_rejects_a_violation_in_any_component() {
    let horizon = 5.0;
    assert_eq!(
        multivariate::Observation::new(&[vec![1.0, 2.0], vec![3.0, 2.5]], horizon).map(|_| ()),
        Err(Error::UnsortedEvents {
            index: 1,
            previous_index: 0,
            previous: 3.0,
            current: 2.5,
        })
    );
    let past = one_ulp_above(horizon);
    assert_eq!(
        multivariate::Observation::new(&[vec![1.0], vec![past]], horizon).map(|_| ()),
        Err(Error::EventOutsideWindow {
            index: 0,
            time: past,
            horizon,
        })
    );
    assert_eq!(
        multivariate::Observation::new(&[vec![], vec![-0.5, 1.0]], horizon).map(|_| ()),
        Err(Error::EventOutsideWindow {
            index: 0,
            time: -0.5,
            horizon,
        })
    );
    assert_eq!(
        multivariate::Observation::new(&[vec![1.0], vec![2.0, f64::NAN]], horizon).map(|_| ()),
        Err(Error::NonFiniteEvent { index: 1 })
    );
}

/// `edge_case_hand_calculations.md` C.
#[test]
fn a_two_component_event_at_the_horizon_matches_the_hand_calculation() {
    let parameters =
        multivariate::Parameters::new(vec![0.8, 0.4], vec![0.3, 0.1, 0.2, 0.25], 1.7).unwrap();
    let events = vec![vec![5.0], vec![]];
    let observation = multivariate::Observation::new(&events, 5.0).unwrap();

    let value = multivariate::negative_log_likelihood(&parameters, &observation).unwrap();
    assert!(close(value, 6.223143551314211), "nll {value:?}");

    let (again, gradient) =
        multivariate::negative_log_likelihood_and_gradient(&parameters, &observation).unwrap();
    assert_eq!(value.to_bits(), again.to_bits());
    assert!(close(gradient.baseline[0], 3.75), "{gradient:?}");
    assert!(close(gradient.baseline[1], 5.0), "{gradient:?}");
    for entry in &gradient.excitation {
        assert!(entry.abs() <= HAND_CALCULATION_TOLERANCE, "{gradient:?}");
    }
    assert!(
        gradient.decay.abs() <= HAND_CALCULATION_TOLERANCE,
        "{gradient:?}"
    );
}

/// `edge_case_hand_calculations.md` E: the cross-component tie contributes nothing to
/// the compensator of the other component at that instant.
#[test]
fn the_multivariate_compensator_with_a_cross_component_tie_matches_the_hand_calculation() {
    let parameters =
        multivariate::Parameters::new(vec![0.2, 0.5], vec![0.1, 0.6, 0.05, 0.15], 1.2).unwrap();
    let events = vec![vec![1.0, 2.5], vec![2.5, 4.0]];
    let observation = multivariate::Observation::new(&events, 6.0).unwrap();
    let actual = multivariate::compensator_at_events(&parameters, &observation).unwrap();
    let expected = [
        vec![0.2, 0.5834701111778413],
        vec![1.2917350555889207, 2.215574036233318],
    ];
    for i in 0..2 {
        assert_eq!(actual[i].len(), expected[i].len());
        for k in 0..expected[i].len() {
            assert!(
                close(actual[i][k], expected[i][k]),
                "Lambda_{i}(t_{k}): got {:?}, hand calculation {:?}",
                actual[i][k],
                expected[i][k]
            );
        }
    }
}

// ------------------------------------------------------------------- fit, fit_from

#[test]
fn univariate_fit_rejects_fewer_than_three_events() {
    for times in [&[][..], &[1.0][..], &[1.0, 2.0][..]] {
        let observation = univariate::Observation::new(times, 5.0).unwrap();
        assert!(
            matches!(
                univariate::fit(&observation),
                Err(Error::InsufficientData { events }) if events == times.len()
            ),
            "{times:?}"
        );
    }
    // Three is enough to attempt: the bound is documented as deliberately low.
    let observation = univariate::Observation::new(&[1.0, 2.0, 3.0], 5.0).unwrap();
    assert!(univariate::fit(&observation).is_ok());
}

#[test]
fn multivariate_fit_rejects_fewer_than_d_plus_d_squared_plus_one_events() {
    // d = 2 needs 7 events in total, however they are distributed.
    let six = vec![vec![0.5, 1.5, 2.5], vec![1.0, 2.0, 3.0]];
    let observation = multivariate::Observation::new(&six, 5.0).unwrap();
    assert_eq!(
        multivariate::fit(&observation).map(|_| ()),
        Err(Error::InsufficientData { events: 6 })
    );
    let seven = vec![vec![0.5, 1.5, 2.5], vec![1.0, 2.0, 3.0, 4.0]];
    let observation = multivariate::Observation::new(&seven, 5.0).unwrap();
    assert!(multivariate::fit(&observation).is_ok());
}

#[test]
fn multivariate_fit_from_rejects_a_start_of_the_wrong_length() {
    let events = vec![vec![0.5, 1.5, 2.5, 3.5], vec![1.0, 2.0, 3.0, 4.0]];
    let observation = multivariate::Observation::new(&events, 5.0).unwrap();
    for wrong in [0usize, 6, 8] {
        assert_eq!(
            multivariate::fit_from(&observation, vec![0.0; wrong]).map(|_| ()),
            Err(Error::DimensionMismatch {
                what: "start",
                actual: wrong,
                expected: 7,
                dimension: 2,
            })
        );
    }
    let six = vec![vec![0.5, 1.5, 2.5], vec![1.0, 2.0, 3.0]];
    let observation = multivariate::Observation::new(&six, 5.0).unwrap();
    assert_eq!(
        multivariate::fit_from(&observation, vec![0.0; 7]).map(|_| ()),
        Err(Error::InsufficientData { events: 6 })
    );
}

// --------------------------------------------------------------------------- simulate

#[test]
fn simulate_rejects_a_bad_horizon() {
    let uni = univariate::Parameters::new(0.5, 0.4, 1.0).unwrap();
    let multi = multivariate::Parameters::new(vec![0.5, 0.4], vec![0.1; 4], 1.0).unwrap();
    for horizon in [0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(
            univariate::simulate(&uni, horizon, &mut rng),
            Err(Error::InvalidHorizon { horizon })
        );
        assert_eq!(
            multivariate::simulate(&multi, horizon, &mut rng),
            Err(Error::InvalidHorizon { horizon })
        );
    }
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    assert!(matches!(
        univariate::simulate(&uni, f64::NAN, &mut rng),
        Err(Error::InvalidHorizon { horizon }) if horizon.is_nan()
    ));
    assert!(matches!(
        multivariate::simulate(&multi, f64::NAN, &mut rng),
        Err(Error::InvalidHorizon { horizon }) if horizon.is_nan()
    ));
}

// ------------------------------------------------------------- randomised sweeps

/// One of the violations a sweep injects. Each is chosen so that it breaks exactly one
/// rule of C8, whatever the surrounding data.
#[derive(Debug, Clone, Copy)]
enum Violation {
    /// `NaN` at a random index: not finite, and comparisons with it are all false, so
    /// the window and order rules are untouched.
    NotANumber,
    /// One ulp past the horizon, at the **last** index, so order is untouched.
    PastTheHorizon,
    /// A negative value at index **0**, so order is untouched.
    Negative,
    /// A value below its predecessor, inside the window, at a random index above 0.
    Unsorted,
}

fn violation() -> impl Strategy<Value = Violation> {
    prop_oneof![
        Just(Violation::NotANumber),
        Just(Violation::PastTheHorizon),
        Just(Violation::Negative),
        Just(Violation::Unsorted),
    ]
}

/// Applies `violation` to `times` and returns the error the contract requires, or
/// `None` when the data cannot host that violation (an unsorted pair needs a positive
/// predecessor).
fn inject(
    times: &mut [f64],
    horizon: f64,
    violation: Violation,
    position: usize,
    magnitude: f64,
) -> Option<Error> {
    let n = times.len();
    match violation {
        Violation::NotANumber => {
            let index = position % n;
            times[index] = f64::NAN;
            Some(Error::NonFiniteEvent { index })
        }
        Violation::PastTheHorizon => {
            let time = one_ulp_above(horizon) * (1.0 + magnitude);
            times[n - 1] = time;
            Some(Error::EventOutsideWindow {
                index: n - 1,
                time,
                horizon,
            })
        }
        Violation::Negative => {
            let time = -(magnitude + f64::MIN_POSITIVE);
            times[0] = time;
            Some(Error::EventOutsideWindow {
                index: 0,
                time,
                horizon,
            })
        }
        Violation::Unsorted => {
            if n < 2 {
                return None;
            }
            let index = 1 + position % (n - 1);
            let previous = times[index - 1];
            if previous <= 0.0 {
                return None;
            }
            // Strictly below the predecessor, still inside `[0, T]`.
            let current = previous * (0.5 * magnitude.min(1.0));
            if current >= previous {
                return None;
            }
            times[index] = current;
            Some(Error::UnsortedEvents {
                index,
                previous_index: index - 1,
                previous,
                current,
            })
        }
    }
}

/// Compares two errors, treating a `NaN` field as equal to a `NaN` field.
fn same_error(actual: &Error, expected: &Error) -> bool {
    match (actual, expected) {
        (Error::NonFiniteEvent { index: a }, Error::NonFiniteEvent { index: e }) => a == e,
        _ => actual == expected,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn univariate_parameters_accept_positive_finite_and_reject_one_bad_entry(
        baseline in 1e-9f64..1e6,
        excitation in 1e-9f64..1e6,
        decay in 1e-9f64..1e6,
        slot in 0usize..3,
        bad in prop_oneof![
            Just(0.0f64),
            (-1e6f64..0.0),
            Just(f64::NAN),
            Just(f64::INFINITY),
            Just(f64::NEG_INFINITY),
        ],
    ) {
        let parameters = univariate::Parameters::new(baseline, excitation, decay).unwrap();
        prop_assert_eq!(parameters.baseline().to_bits(), baseline.to_bits());
        prop_assert_eq!(parameters.excitation().to_bits(), excitation.to_bits());
        prop_assert_eq!(parameters.decay().to_bits(), decay.to_bits());

        let mut values = [baseline, excitation, decay];
        values[slot] = bad;
        let name = ["baseline", "excitation", "decay"][slot];
        let error = univariate::Parameters::new(values[0], values[1], values[2])
            .expect_err("one bad entry must be rejected");
        if bad.is_finite() {
            prop_assert_eq!(error, Error::NonPositiveParameter { name, value: bad });
        } else {
            prop_assert!(matches!(
                error,
                Error::NonFiniteParameter { name: got, value }
                    if got == name && (value.is_nan() || value == bad)
            ), "{:?}", error);
        }
    }

    #[test]
    fn univariate_observation_accepts_random_valid_realizations(
        horizon in 0.5f64..100.0,
        raw in prop::collection::vec(0.0f64..=1.0, 0..80),
        on_grid in prop::bool::ANY,
    ) {
        let times = sorted_times(&raw, horizon, on_grid);
        let observation = univariate::Observation::new(&times, horizon).unwrap();
        prop_assert_eq!(observation.len(), times.len());
        prop_assert_eq!(observation.is_empty(), times.is_empty());
        prop_assert_eq!(observation.times(), &times[..]);
        prop_assert_eq!(observation.horizon().to_bits(), horizon.to_bits());
    }

    #[test]
    fn univariate_observation_rejects_one_injected_violation(
        horizon in 0.5f64..100.0,
        raw in prop::collection::vec(0.0f64..=1.0, 1..60),
        on_grid in prop::bool::ANY,
        violation in violation(),
        position in any::<usize>(),
        magnitude in 0.0f64..10.0,
    ) {
        let mut times = sorted_times(&raw, horizon, on_grid);
        let expected = inject(&mut times, horizon, violation, position, magnitude);
        prop_assume!(expected.is_some());
        let expected = expected.unwrap();
        let actual = univariate::Observation::new(&times, horizon)
            .expect_err("an injected violation must be rejected");
        prop_assert!(same_error(&actual, &expected),
            "{:?}: got {:?}, contract requires {:?}", violation, actual, expected);
    }

    #[test]
    fn univariate_observation_rejects_a_random_bad_horizon(
        horizon in prop_oneof![(-1e6f64..=0.0), Just(0.0f64), Just(f64::INFINITY), Just(f64::NAN)],
        raw in prop::collection::vec(0.0f64..=1.0, 0..10),
    ) {
        let times = sorted_times(&raw, 1.0, false);
        let error = univariate::Observation::new(&times, horizon)
            .expect_err("a non-positive or non-finite horizon must be rejected");
        prop_assert!(matches!(
            error,
            Error::InvalidHorizon { horizon: got } if got.is_nan() == horizon.is_nan()
                && (got.is_nan() || got == horizon)
        ), "{:?}", error);
    }

    /// The randomised form of case A: with the only event at `T`, the value is
    /// `mu*T - ln(mu)` and the excitation and decay gradients vanish, for every
    /// parameter set.
    #[test]
    fn an_event_at_exactly_the_horizon_is_mu_t_minus_ln_mu(
        baseline in 0.05f64..5.0,
        excitation in 0.01f64..0.95,
        decay in 0.05f64..5.0,
        horizon in 0.5f64..50.0,
    ) {
        let parameters = univariate::Parameters::new(baseline, excitation, decay).unwrap();
        let times = [horizon];
        let observation = univariate::Observation::new(&times, horizon).unwrap();
        let expected = baseline * horizon - baseline.ln();
        let scale = baseline * horizon + baseline.ln().abs();
        let (value, gradient) =
            univariate::negative_log_likelihood_and_gradient(&parameters, &observation);
        prop_assert!((value - expected).abs() <= HAND_CALCULATION_TOLERANCE * scale,
            "nll {} vs mu*T - ln(mu) = {}", value, expected);
        prop_assert!((gradient.baseline - (horizon - 1.0 / baseline)).abs()
            <= HAND_CALCULATION_TOLERANCE * horizon.max(1.0 / baseline));
        prop_assert!(gradient.excitation.abs() <= HAND_CALCULATION_TOLERANCE);
        prop_assert!(gradient.decay.abs() <= HAND_CALCULATION_TOLERANCE);
    }

    #[test]
    fn multivariate_parameters_accept_valid_and_reject_one_bad_entry(
        d in 1usize..=5,
        seed in 0u64..100_000,
        kind in 0u8..6,
        bad_finite in -1e3f64..=0.0,
        use_nan in prop::bool::ANY,
    ) {
        let mut rng = common::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let baseline: Vec<f64> = (0..d).map(|_| 1e-6 + rng.next_f64() * 4.0).collect();
        // Zero entries included: they are legitimate.
        let excitation: Vec<f64> = (0..d * d)
            .map(|_| if rng.next_f64() < 0.2 { 0.0 } else { rng.next_f64() })
            .collect();
        let decay = 1e-6 + rng.next_f64() * 6.0;

        let parameters =
            multivariate::Parameters::new(baseline.clone(), excitation.clone(), decay).unwrap();
        prop_assert_eq!(parameters.dimension(), d);
        prop_assert_eq!(parameters.baseline(), &baseline[..]);
        prop_assert_eq!(parameters.excitation(), &excitation[..]);
        for i in 0..d {
            for j in 0..d {
                // Row-major, "j excites i" (C6).
                prop_assert_eq!(
                    parameters.excitation_at(i, j).to_bits(),
                    excitation[i * d + j].to_bits()
                );
            }
        }

        let position = rng.next_usize(d * d);
        let bad = if use_nan { f64::NAN } else { bad_finite };
        let (mut baseline, mut excitation, mut decay) = (baseline, excitation, decay);
        let expected: Error = match kind {
            0 => {
                let k = position % d;
                baseline[k] = bad;
                if use_nan {
                    Error::NonFiniteParameter { name: "baseline", value: f64::NAN }
                } else {
                    Error::NonPositiveParameter { name: "baseline", value: bad }
                }
            }
            1 => {
                // Zero is legitimate for excitation, so the finite bad value is
                // strictly negative here.
                let value = if use_nan { f64::NAN } else { bad_finite - f64::MIN_POSITIVE };
                excitation[position] = value;
                Error::InvalidExcitation { row: position / d, column: position % d, value }
            }
            2 => {
                decay = bad;
                if use_nan {
                    Error::NonFiniteParameter { name: "decay", value: f64::NAN }
                } else {
                    Error::NonPositiveParameter { name: "decay", value: bad }
                }
            }
            3 => {
                excitation.push(0.1);
                Error::DimensionMismatch {
                    what: "excitation", actual: d * d + 1, expected: d * d, dimension: d,
                }
            }
            4 => {
                excitation.pop();
                Error::DimensionMismatch {
                    what: "excitation", actual: d * d - 1, expected: d * d, dimension: d,
                }
            }
            _ => {
                baseline.clear();
                excitation.clear();
                Error::EmptyProcess
            }
        };
        let actual = multivariate::Parameters::new(baseline, excitation, decay)
            .expect_err("one bad entry must be rejected");
        let agrees = match (&actual, &expected) {
            (
                Error::NonFiniteParameter { name: a, value: av },
                Error::NonFiniteParameter { name: e, value: ev },
            ) => a == e && av.is_nan() && ev.is_nan(),
            (
                Error::InvalidExcitation { row: ar, column: ac, value: av },
                Error::InvalidExcitation { row: er, column: ec, value: ev },
            ) => ar == er && ac == ec && (av.to_bits() == ev.to_bits() || (av.is_nan() && ev.is_nan())),
            _ => actual == expected,
        };
        prop_assert!(agrees, "kind {}: got {:?}, contract requires {:?}", kind, actual, expected);
    }

    #[test]
    fn multivariate_observation_accepts_random_valid_realizations(
        d in 1usize..=4,
        horizon in 0.5f64..100.0,
        raw in prop::collection::vec(prop::collection::vec(0.0f64..=1.0, 0..30), 4),
        on_grid in prop::bool::ANY,
    ) {
        let events: Vec<Vec<f64>> = raw
            .iter()
            .take(d)
            .map(|component| sorted_times(component, horizon, on_grid))
            .collect();
        let observation = multivariate::Observation::new(&events, horizon).unwrap();
        prop_assert_eq!(observation.dimension(), d);
        prop_assert_eq!(observation.len(), events.iter().map(Vec::len).sum::<usize>());
        prop_assert_eq!(observation.is_empty(), events.iter().all(Vec::is_empty));
        prop_assert_eq!(observation.events(), &events[..]);
    }

    #[test]
    fn multivariate_observation_rejects_one_injected_violation(
        d in 1usize..=4,
        horizon in 0.5f64..100.0,
        raw in prop::collection::vec(prop::collection::vec(0.0f64..=1.0, 1..30), 4),
        on_grid in prop::bool::ANY,
        component in 0usize..4,
        violation in violation(),
        position in any::<usize>(),
        magnitude in 0.0f64..10.0,
    ) {
        let mut events: Vec<Vec<f64>> = raw
            .iter()
            .take(d)
            .map(|component| sorted_times(component, horizon, on_grid))
            .collect();
        let component = component % d;
        let expected = inject(&mut events[component], horizon, violation, position, magnitude);
        prop_assume!(expected.is_some());
        let expected = expected.unwrap();
        let actual = multivariate::Observation::new(&events, horizon)
            .expect_err("an injected violation must be rejected");
        prop_assert!(same_error(&actual, &expected),
            "{:?} in component {}: got {:?}, contract requires {:?}",
            violation, component, actual, expected);
    }

    /// Randomised counterpart of the three fixed mismatch cases in
    /// `multivariate_loglikelihood.rs`: every entry point, both directions, every
    /// pair of dimensions up to 4.
    #[test]
    fn every_entry_point_rejects_a_random_mismatch(
        parameter_dimension in 1usize..=4,
        observation_dimension in 1usize..=4,
        seed in 0u64..100_000,
    ) {
        let mut rng = common::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let dp = parameter_dimension;
        let parameters = multivariate::Parameters::new(
            (0..dp).map(|_| 0.1 + rng.next_f64()).collect(),
            (0..dp * dp).map(|_| rng.next_f64() * 0.5 / dp as f64).collect(),
            0.5 + rng.next_f64(),
        ).unwrap();
        let events: Vec<Vec<f64>> = (0..observation_dimension)
            .map(|_| {
                let n = rng.next_usize(6);
                let raw: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
                sorted_times(&raw, 5.0, false)
            })
            .collect();
        let observation = multivariate::Observation::new(&events, 5.0).unwrap();

        if dp == observation_dimension {
            prop_assert!(multivariate::negative_log_likelihood(&parameters, &observation).is_ok());
            prop_assert!(multivariate::negative_log_likelihood_and_gradient(&parameters, &observation).is_ok());
            prop_assert!(multivariate::compensator_at_events(&parameters, &observation).is_ok());
            #[cfg(feature = "rayon")]
            prop_assert!(multivariate::negative_log_likelihood_parallel(&parameters, &observation).is_ok());
        } else {
            let expected = Error::ProcessDimensionMismatch {
                parameters: dp,
                observation: observation_dimension,
            };
            prop_assert_eq!(
                multivariate::negative_log_likelihood(&parameters, &observation).unwrap_err(),
                expected.clone()
            );
            prop_assert_eq!(
                multivariate::negative_log_likelihood_and_gradient(&parameters, &observation).unwrap_err(),
                expected.clone()
            );
            prop_assert_eq!(
                multivariate::compensator_at_events(&parameters, &observation).unwrap_err(),
                expected.clone()
            );
            #[cfg(feature = "rayon")]
            prop_assert_eq!(
                multivariate::negative_log_likelihood_parallel(&parameters, &observation).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn univariate_fit_rejects_any_realization_with_fewer_than_three_events(
        n in 0usize..3,
        raw in prop::collection::vec(0.0f64..=1.0, 3),
        horizon in 1.0f64..50.0,
    ) {
        let times = sorted_times(&raw[..n], horizon, false);
        let observation = univariate::Observation::new(&times, horizon).unwrap();
        prop_assert!(
            matches!(
                univariate::fit(&observation),
                Err(Error::InsufficientData { events }) if events == n
            ),
            "n = {}", n
        );
        // With three, the data bound is satisfied whatever the optimizer then does.
        let three = sorted_times(&raw, horizon, false);
        let observation = univariate::Observation::new(&three, horizon).unwrap();
        prop_assert!(
            !matches!(
                univariate::fit(&observation),
                Err(Error::InsufficientData { .. })
            ),
            "three events satisfy the data bound"
        );
    }

    #[test]
    fn multivariate_fit_rejects_any_realization_below_the_parameter_count(
        d in 1usize..=3,
        seed in 0u64..100_000,
        deficit in 1usize..=6,
    ) {
        let needed = d + d * d + 1;
        let n = needed.saturating_sub(deficit);
        let mut rng = common::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        // Distribute `n` events over the components at random.
        let mut events: Vec<Vec<f64>> = vec![Vec::new(); d];
        for _ in 0..n {
            events[rng.next_usize(d)].push(rng.next_f64() * 5.0);
        }
        for component in &mut events {
            component.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let observation = multivariate::Observation::new(&events, 5.0).unwrap();
        prop_assert_eq!(
            multivariate::fit(&observation).map(|_| ()),
            Err(Error::InsufficientData { events: n })
        );
        prop_assert_eq!(
            multivariate::fit_from(&observation, vec![0.0; needed]).map(|_| ()),
            Err(Error::InsufficientData { events: n })
        );
    }

    #[test]
    fn multivariate_fit_from_rejects_any_start_of_the_wrong_length(
        d in 1usize..=3,
        seed in 0u64..100_000,
        wrong in 0usize..20,
    ) {
        let needed = d + d * d + 1;
        prop_assume!(wrong != needed);
        let mut rng = common::Lcg::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut events: Vec<Vec<f64>> = vec![Vec::new(); d];
        for _ in 0..needed + 2 {
            events[rng.next_usize(d)].push(rng.next_f64() * 5.0);
        }
        for component in &mut events {
            component.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let observation = multivariate::Observation::new(&events, 5.0).unwrap();
        prop_assert_eq!(
            multivariate::fit_from(&observation, vec![0.0; wrong]).map(|_| ()),
            Err(Error::DimensionMismatch {
                what: "start",
                actual: wrong,
                expected: needed,
                dimension: d,
            })
        );
    }

    #[test]
    fn simulate_rejects_any_bad_horizon_and_honours_any_good_one(
        bad in prop_oneof![(-1e6f64..=0.0), Just(0.0f64), Just(f64::INFINITY), Just(f64::NAN)],
        good in 0.5f64..200.0,
        baseline in 0.1f64..2.0,
        excitation in 0.05f64..0.9,
        decay in 0.2f64..4.0,
        seed in 0u64..100_000,
    ) {
        let uni = univariate::Parameters::new(baseline, excitation, decay).unwrap();
        let multi = multivariate::Parameters::new(
            vec![baseline, 0.5 * baseline],
            vec![0.5 * excitation; 4],
            decay,
        ).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let error = univariate::simulate(&uni, bad, &mut rng).expect_err("bad horizon");
        prop_assert!(
            matches!(error, Error::InvalidHorizon { horizon }
                if horizon.is_nan() == bad.is_nan() && (horizon.is_nan() || horizon == bad)),
            "univariate: {:?}", error
        );
        let error = multivariate::simulate(&multi, bad, &mut rng).expect_err("bad horizon");
        prop_assert!(
            matches!(error, Error::InvalidHorizon { horizon }
                if horizon.is_nan() == bad.is_nan() && (horizon.is_nan() || horizon == bad)),
            "multivariate: {:?}", error
        );

        // Randomised counterpart of `simulated_realizations_satisfy_the_input_contract`
        // in `simulator.rs` and `multivariate_simulator.rs`.
        let times = univariate::simulate(&uni, good, &mut rng).unwrap();
        univariate::Observation::new(&times, good).unwrap();
        prop_assert!(times.windows(2).all(|w| w[0] < w[1]));

        let events = multivariate::simulate(&multi, good, &mut rng).unwrap();
        multivariate::Observation::new(&events, good).unwrap();
        for component in &events {
            prop_assert!(component.windows(2).all(|w| w[0] < w[1]));
        }
    }
}

/// The message is what a caller sees; it is pinned once so a formatting slip in the
/// `#[error]` string — the ten-space run that #51 fixed — is caught.
#[test]
fn a_dimension_mismatch_renders_single_spaced() {
    let error = multivariate::Parameters::new(vec![0.5, 0.5], vec![0.1; 3], 1.0).unwrap_err();
    assert_eq!(
        error.to_string(),
        "dimension mismatch: excitation has length 3, expected 4 for a 2-component process"
    );
}
