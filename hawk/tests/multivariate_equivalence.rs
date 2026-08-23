//! `d = 1` equivalence between the multivariate and univariate paths (M2 Part B
//! step 9).
//!
//! One assertion, and it catches most of the ways a generalisation goes wrong: on the
//! same events and parameters, `multivariate` at `d = 1` must return **bitwise** the
//! same negative log-likelihood and gradient as `univariate`.
//!
//! # Why bitwise
//!
//! The multivariate expression reduces to the univariate one symbolically
//! (`multivariate_loglikelihood.md` §6), so any difference at all is a difference in
//! the operations performed, which is exactly what a generalisation is at risk of
//! introducing. A tolerance would accept a reordered sum, a regrouped product, or an
//! accumulator that multiplies by a tie count instead of adding — all of which are
//! wrong in the same way and none of which a `1e-12` gate would notice.
//!
//! Two rules in `multivariate_loglikelihood.md` §5.1 exist solely to make this hold.
//! Written the obvious way instead, the Part A check failed 134 of 600 cases.
//!
//! # Sabotage
//!
//! Replacing the per-event accumulation with `count * value` turned
//! `nll_agrees_on_tied_input` red while leaving the tie-free cases green. Regrouping
//! the intensity product as `beta * (alpha * state)` turned the randomized case red.
//! Dropping the `S[i][j]` factoring in favour of the natural `d/dbeta` assembly turned
//! only the gradient assertions red. Recorded in `docs/verification-log.md`.

mod common;

use hawk::multivariate;
use hawk::univariate;
use proptest::prelude::*;

fn build(times: &[f64], horizon: f64, baseline: f64, excitation: f64, decay: f64) -> (f64, f64) {
    let uni_parameters = univariate::Parameters::new(baseline, excitation, decay).unwrap();
    let uni_observation = univariate::Observation::new(times, horizon).unwrap();
    let uni = univariate::negative_log_likelihood(&uni_parameters, &uni_observation);

    let events = vec![times.to_vec()];
    let multi_parameters =
        multivariate::Parameters::new(vec![baseline], vec![excitation], decay).unwrap();
    let multi_observation = multivariate::Observation::new(&events, horizon).unwrap();
    let multi = multivariate::negative_log_likelihood(&multi_parameters, &multi_observation);
    (uni, multi)
}

fn assert_nll_identical(times: &[f64], horizon: f64, p: (f64, f64, f64), context: &str) {
    let (uni, multi) = build(times, horizon, p.0, p.1, p.2);
    assert_eq!(
        uni.to_bits(),
        multi.to_bits(),
        "{context}: univariate {uni:?} (bits {:#018x}) vs multivariate at d=1 \
         {multi:?} (bits {:#018x}); difference {:e}",
        uni.to_bits(),
        multi.to_bits(),
        (uni - multi).abs()
    );
}

fn assert_gradient_identical(times: &[f64], horizon: f64, p: (f64, f64, f64), context: &str) {
    let uni_parameters = univariate::Parameters::new(p.0, p.1, p.2).unwrap();
    let uni_observation = univariate::Observation::new(times, horizon).unwrap();
    let (uni_value, uni_gradient) =
        univariate::negative_log_likelihood_and_gradient(&uni_parameters, &uni_observation);

    let events = vec![times.to_vec()];
    let multi_parameters = multivariate::Parameters::new(vec![p.0], vec![p.1], p.2).unwrap();
    let multi_observation = multivariate::Observation::new(&events, horizon).unwrap();
    let (multi_value, multi_gradient) =
        multivariate::negative_log_likelihood_and_gradient(&multi_parameters, &multi_observation);

    assert_eq!(
        uni_value.to_bits(),
        multi_value.to_bits(),
        "{context}: value from the gradient path differs"
    );
    for (name, u, m) in [
        (
            "baseline",
            uni_gradient.baseline,
            multi_gradient.baseline[0],
        ),
        (
            "excitation",
            uni_gradient.excitation,
            multi_gradient.excitation[0],
        ),
        ("decay", uni_gradient.decay, multi_gradient.decay),
    ] {
        assert_eq!(
            u.to_bits(),
            m.to_bits(),
            "{context}: d/d{name} univariate {u:?} vs multivariate {m:?}, \
             difference {:e}",
            (u - m).abs()
        );
    }
}

#[test]
fn nll_agrees_on_degenerate_input() {
    for (times, horizon) in [
        (vec![], 5.0),
        (vec![2.5], 5.0),
        (vec![0.0], 5.0),
        (vec![5.0], 5.0),
        (vec![0.0, 5.0], 5.0),
    ] {
        assert_nll_identical(&times, horizon, (0.9, 0.4, 1.7), &format!("{times:?}"));
        assert_gradient_identical(&times, horizon, (0.9, 0.4, 1.7), &format!("{times:?}"));
    }
}

/// Ties are where the two accumulation rules of §5.1 bite.
#[test]
fn nll_agrees_on_tied_input() {
    for (times, horizon) in [
        (vec![1.0, 2.0, 2.0, 3.0], 5.0),
        (vec![1.0, 2.0, 2.0, 2.0, 3.5], 6.0),
        (vec![0.0, 0.0, 1.5, 3.0, 5.0, 5.0], 5.0),
        (vec![2.0, 2.0, 2.0, 2.0], 4.0),
        // Multiplicities where `count * x` and `x` added `count` times differ in
        // f64, which is what forces the per-event accumulation rule of
        // `multivariate_loglikelihood.md` §5.1.
        //
        // Which multiplicities discriminate depends on the value being accumulated,
        // and that is not obvious: a sabotage replacing per-event accumulation with
        // `count * value` survived a multiplicity-7 case and died on a multiplicity-6
        // one. Several are used rather than reasoning about which.
        (vec![1.0; 4], 4.0),
        (vec![1.0; 5], 4.0),
        (vec![1.0; 6], 4.0),
        (vec![1.0; 7], 4.0),
        (vec![1.0; 9], 4.0),
        (vec![2.0; 6], 5.0),
        (vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0], 4.0),
    ] {
        let label = format!("{times:?}");
        assert_nll_identical(&times, horizon, (0.7, 0.5, 1.3), &label);
        assert_gradient_identical(&times, horizon, (0.7, 0.5, 1.3), &label);
    }
}

#[test]
fn nll_agrees_on_events_packed_against_the_horizon() {
    let horizon = 1.0;
    for times in [
        vec![horizon - 1e-11, horizon - 1e-12, horizon - 1e-13],
        vec![horizon - 1e-13, horizon - 1e-14, horizon],
    ] {
        for p in [(0.5, 0.6, 1.0), (2.0, 0.9, 5.0)] {
            let label = format!("{times:?} at {p:?}");
            assert_nll_identical(&times, horizon, p, &label);
            assert_gradient_identical(&times, horizon, p, &label);
        }
    }
}

#[test]
fn nll_agrees_on_long_realizations() {
    let mut rng = common::Lcg::new(0x2545_F491_4F6C_DD1D);
    for (count, decay) in [(2_000usize, 0.05), (2_000, 4.0), (20_000, 1.0)] {
        let horizon = count as f64 / 2.0;
        let mut times: Vec<f64> = (0..count).map(|_| rng.next_f64() * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let label = format!("n={count} beta={decay}");
        assert_nll_identical(&times, horizon, (1.3, 0.6, decay), &label);
        assert_gradient_identical(&times, horizon, (1.3, 0.6, decay), &label);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn agrees_over_random_parameters_and_events(
        baseline in 0.01f64..8.0,
        excitation in 0.001f64..0.99,
        decay in 0.01f64..8.0,
        horizon in 1.0f64..200.0,
        raw in prop::collection::vec(0.0f64..1.0, 0..150),
    ) {
        let mut times: Vec<f64> = raw.iter().map(|u| u * horizon).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (uni, multi) = build(&times, horizon, baseline, excitation, decay);
        prop_assert_eq!(uni.to_bits(), multi.to_bits(),
            "univariate {:?} vs multivariate {:?}", uni, multi);
    }

    #[test]
    fn agrees_when_ties_are_common(
        baseline in 0.05f64..3.0,
        excitation in 0.01f64..0.95,
        decay in 0.05f64..5.0,
        grid in prop::collection::vec(0u32..8, 1..50),
    ) {
        let horizon = 9.0;
        let mut times: Vec<f64> = grid.iter().map(|g| f64::from(*g)).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let uni_parameters = univariate::Parameters::new(baseline, excitation, decay).unwrap();
        let uni_observation = univariate::Observation::new(&times, horizon).unwrap();
        let (uv, ug) = univariate::negative_log_likelihood_and_gradient(
            &uni_parameters, &uni_observation);

        let events = vec![times.clone()];
        let mp = multivariate::Parameters::new(vec![baseline], vec![excitation], decay).unwrap();
        let mo = multivariate::Observation::new(&events, horizon).unwrap();
        let (mv, mg) = multivariate::negative_log_likelihood_and_gradient(&mp, &mo);

        prop_assert_eq!(uv.to_bits(), mv.to_bits());
        prop_assert_eq!(ug.baseline.to_bits(), mg.baseline[0].to_bits());
        prop_assert_eq!(ug.excitation.to_bits(), mg.excitation[0].to_bits());
        prop_assert_eq!(ug.decay.to_bits(), mg.decay.to_bits());
    }
}
