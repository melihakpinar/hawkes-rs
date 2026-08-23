//! Oracles for the multivariate simulator (M2 Part B step 7).
//!
//! The two CLAUDE.md §3 oracles that are anchored outside the implementation, applied
//! **per component**:
//!
//! 1. **Analytic identity.** `Lambda = (I - alpha)^{-1} mu`
//!    (`multivariate_loglikelihood.md` (M7.1), [Bacry2015, Prop. 4 eq. 21]). Each
//!    component's realized rate must converge to its own entry.
//! 2. **Time rescaling.** Each component's compensated times form a unit-rate Poisson
//!    process [Laub2015, Theorem 4]. KS per component.
//!
//! Both are per component on purpose. A pooled mean-intensity test would let an error
//! that moves activity from one component to another cancel exactly; the total would
//! be right and the process wrong. That is not hypothetical — mixing up the excitation
//! orientation does exactly that on a matrix with equal column sums.
//!
//! # Sabotage
//!
//! Removing the spectral-radius check from `stationary_mean_intensity` turned
//! `non_stationary_parameters_have_no_mean_intensity` red: `I - alpha` is still
//! invertible for the matrix used there, so the solve succeeds and returns a vector
//! with negative entries. Transposing the excitation in the simulator turned the
//! per-component mean-intensity test red while leaving the pooled total green.
//! Recorded in `docs/verification-log.md`.

use hawk::multivariate::{
    Observation, Parameters, compensator_at_events, negative_log_likelihood, simulate,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const SEEDS: [u64; 6] = [1, 2, 3, 4, 5, 6];

/// Relative tolerance for a component's mean intensity.
///
/// Same reasoning as the univariate test: the count is an average over clustered
/// events, so the variance is inflated over Poisson by roughly `1/(1-rho)^2`. With the
/// horizons below each component sees thousands of events. 8% rather than the
/// univariate 5% because a single component receives a fraction of the total activity,
/// so its own count is smaller and its relative error correspondingly larger.
const MEAN_INTENSITY_TOLERANCE: f64 = 0.08;

fn ks_statistic_against_unit_exponential(sample: &mut [f64]) -> f64 {
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sample.len() as f64;
    let mut deviation: f64 = 0.0;
    for (index, &value) in sample.iter().enumerate() {
        let theoretical = 1.0 - (-value).exp();
        deviation = deviation.max((theoretical - index as f64 / n).abs());
        deviation = deviation.max(((index as f64 + 1.0) / n - theoretical).abs());
    }
    deviation
}

fn asymmetric_three() -> Parameters {
    // Column sums are deliberately unequal so a transposition cannot preserve the
    // per-component rates.
    Parameters::new(
        vec![0.12, 0.20, 0.75],
        vec![0.10, 0.45, 0.05, 0.05, 0.10, 0.30, 0.20, 0.05, 0.10],
        1.4,
    )
    .unwrap()
}

#[test]
fn converges_to_the_per_component_stationary_mean_intensity() {
    let parameters = asymmetric_three();
    let expected = parameters
        .stationary_mean_intensity()
        .expect("the test parameters are stationary");
    let horizon = 20_000.0;

    for seed in SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let events = simulate(&parameters, horizon, &mut rng).unwrap();
        for (component, times) in events.iter().enumerate() {
            let observed = times.len() as f64 / horizon;
            let relative = (observed - expected[component]).abs() / expected[component];
            assert!(
                relative <= MEAN_INTENSITY_TOLERANCE,
                "seed {seed}, component {component}: observed {observed:?} vs \
                 analytic {:?}, relative error {relative:?} > \
                 {MEAN_INTENSITY_TOLERANCE}. Full analytic vector {expected:?}.",
                expected[component]
            );
        }
    }
}

/// A pooled total can be right while every component is wrong. This pins that the
/// per-component test above is not accidentally a total test.
#[test]
fn the_per_component_test_is_stronger_than_the_total() {
    let parameters = asymmetric_three();
    let expected = parameters.stationary_mean_intensity().unwrap();
    // The components genuinely differ, so agreeing on each is a real constraint.
    let smallest = expected.iter().cloned().fold(f64::INFINITY, f64::min);
    let largest = expected.iter().cloned().fold(0.0, f64::max);
    assert!(
        largest / smallest > 1.5,
        "the test parameters have near-equal component rates ({expected:?}), so the \
         per-component assertion would be little stronger than a total"
    );
}

#[test]
fn time_rescaled_residuals_are_unit_exponential_per_component() {
    let cases = [
        (asymmetric_three(), 8000.0),
        (
            Parameters::new(vec![0.6, 0.4], vec![0.20, 0.35, 0.10, 0.25], 2.0).unwrap(),
            8000.0,
        ),
    ];

    for (parameters, horizon) in cases {
        for seed in SEEDS.iter().take(4) {
            let mut rng = ChaCha8Rng::seed_from_u64(*seed);
            let events = simulate(&parameters, horizon, &mut rng).unwrap();
            let observation = Observation::new(&events, horizon).unwrap();
            let compensators = compensator_at_events(&parameters, &observation);

            for (component, values) in compensators.iter().enumerate() {
                assert!(
                    values.len() > 500,
                    "component {component} has only {} events",
                    values.len()
                );
                let mut residuals = Vec::with_capacity(values.len());
                let mut previous = 0.0;
                for &value in values {
                    residuals.push(value - previous);
                    previous = value;
                }
                let n = residuals.len() as f64;
                let statistic = ks_statistic_against_unit_exponential(&mut residuals);
                // 1% level; see the univariate test for why not 5%.
                let critical = 1.628 / n.sqrt();
                assert!(
                    statistic <= critical,
                    "seed {seed}, component {component}: KS statistic {statistic:?} \
                     > critical {critical:?} at n={n}. The simulator and the \
                     compensator disagree about component {component}'s intensity."
                );
            }
        }
    }
}

/// The KS test must be able to reject, per component.
#[test]
fn the_ks_test_rejects_residuals_from_a_transposed_matrix() {
    let truth = asymmetric_three();
    let horizon = 8000.0;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let events = simulate(&truth, horizon, &mut rng).unwrap();
    let observation = Observation::new(&events, horizon).unwrap();

    // Transpose the excitation matrix and rescale with it.
    let d = truth.dimension();
    let mut transposed = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            transposed[i * d + j] = truth.excitation_at(j, i);
        }
    }
    let wrong = Parameters::new(truth.baseline().to_vec(), transposed, truth.decay()).unwrap();
    let compensators = compensator_at_events(&wrong, &observation);

    let mut any_rejected = false;
    for values in &compensators {
        let mut residuals = Vec::with_capacity(values.len());
        let mut previous = 0.0;
        for &value in values {
            residuals.push(value - previous);
            previous = value;
        }
        let n = residuals.len() as f64;
        let statistic = ks_statistic_against_unit_exponential(&mut residuals);
        if statistic > 1.628 / n.sqrt() {
            any_rejected = true;
        }
    }
    assert!(
        any_rejected,
        "residuals computed with a transposed excitation matrix were accepted on \
         every component; the test has no power against the transposition it exists \
         to catch"
    );
}

/// Constraint (b): stationarity is checked, not inferred from the solve.
#[test]
fn non_stationary_parameters_have_no_mean_intensity() {
    // spectral radius sqrt(1.8) = 1.3416..., but det(I - alpha) = 1 - 1.8 = -0.8, so
    // the linear solve succeeds and returns a vector with negative entries.
    let parameters = Parameters::new(vec![0.5, 0.5], vec![0.0, 2.0, 0.9, 0.0], 1.0).unwrap();
    let radius = parameters.branching_ratio_spectral_radius();
    assert!(
        (radius - 1.3416407864998738).abs() < 1e-9,
        "spectral radius of [[0, 2], [0.9, 0]] should be sqrt(1.8), got {radius:?}. \
         Power iteration without the +I shift oscillates between 0.9 and 2 here and \
         never converges."
    );
    assert!(!parameters.is_stationary());
    assert_eq!(
        parameters.stationary_mean_intensity(),
        None,
        "a non-stationary process has no stationary mean intensity, and returning \
         the solve's output would be worse than failing because it looks like an \
         answer"
    );
}

#[test]
fn spectral_radius_matches_hand_calculations() {
    // Eigenvalues 0.3 and -0.05.
    let p = Parameters::new(vec![0.2, 0.5], vec![0.1, 0.6, 0.05, 0.15], 1.0).unwrap();
    assert!((p.branching_ratio_spectral_radius() - 0.3).abs() < 1e-9);
    assert_eq!(
        p.stationary_mean_intensity(),
        Some(vec![0.6394557823129251, 0.6258503401360543]),
        "hand calculation in multivariate_loglikelihood.md §7"
    );

    // Diagonal: the spectral radius is the largest entry.
    let diagonal = Parameters::new(
        vec![1.0; 3],
        vec![0.2, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.4],
        1.0,
    )
    .unwrap();
    assert!((diagonal.branching_ratio_spectral_radius() - 0.7).abs() < 1e-9);

    // The circulant d = 10 fixture: row sums 0.45, so the Perron root is 0.45.
    let d = 10;
    let mut excitation = vec![0.0; d * d];
    for i in 0..d {
        excitation[i * d + i] += 0.05;
        excitation[i * d + (i + 1) % d] += 0.30;
        excitation[i * d + (i + 3) % d] += 0.10;
    }
    let circulant = Parameters::new(vec![0.5; d], excitation, 1.0).unwrap();
    assert!((circulant.branching_ratio_spectral_radius() - 0.45).abs() < 1e-9);
}

#[test]
fn simulated_realizations_satisfy_the_input_contract() {
    let parameters = asymmetric_three();
    for seed in SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let events = simulate(&parameters, 500.0, &mut rng).unwrap();
        Observation::new(&events, 500.0).expect("simulator must produce valid input");
        for (component, times) in events.iter().enumerate() {
            assert!(
                times.windows(2).all(|w| w[0] < w[1]),
                "component {component} has a tie, which should have probability zero"
            );
        }
        // A finite likelihood is a cheap end-to-end check that the realization is
        // consistent with the parameters that produced it.
        let observation = Observation::new(&events, 500.0).unwrap();
        assert!(negative_log_likelihood(&parameters, &observation).is_finite());
    }
}

#[test]
fn simulation_is_reproducible_from_a_seed() {
    let parameters = asymmetric_three();
    let first = simulate(&parameters, 300.0, &mut ChaCha8Rng::seed_from_u64(99)).unwrap();
    let second = simulate(&parameters, 300.0, &mut ChaCha8Rng::seed_from_u64(99)).unwrap();
    assert_eq!(first, second, "same seed must give the same realization");
}
