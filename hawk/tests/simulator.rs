//! Oracles for the simulator (M1 Part B step 6).
//!
//! Two of CLAUDE.md §3's five, and the two that could not exist before there was a
//! simulator:
//!
//! 1. **Analytic identity.** The stationary mean intensity has the closed form
//!    `mu / (1 - alpha)` [Laub2015, eq. 6]. Long realizations must converge to it.
//! 2. **Time rescaling.** By the random time change theorem [Laub2015, Theorem 4],
//!    transforming event times by the compensator yields a unit-rate Poisson process.
//!    KS test.
//!
//! Oracle 2 is the strongest thing in this file: it validates the simulator and the
//! compensator *jointly*. A simulator drawing from the wrong intensity and a
//! compensator integrating the wrong intensity would have to be wrong in precisely
//! matching ways to still produce a unit-rate Poisson process.
//!
//! # Sabotage
//!
//! Removing the `excitation += 1.0` after an accepted event — making the simulator
//! generate a Poisson process instead of a Hawkes one — turned both oracles red.
//! Using `lambda(t+)` *after* decaying the state, rather than before, turned the KS
//! test red while leaving the mean-intensity test green, which is the asymmetry that
//! makes the residual oracle worth having. Recorded in `docs/verification-log.md`.

use hawk::univariate::{Observation, Parameters, compensator_at_events, simulate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Seeds are fixed so a failure is reproducible. They are arbitrary, not chosen: the
/// tests were written, then run, and no seed was changed afterwards. Tuning a seed
/// until a statistical test passes is how a broken oracle gets committed.
const SEEDS: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// Relative tolerance for the stationary mean intensity.
///
/// `N(T)/T` is an average over roughly `T*mu/(1-alpha)` events, but they are
/// *clustered*, so the variance is inflated over the Poisson case by roughly
/// `1/(1-alpha)^2` — the same factor that makes a Hawkes process bursty. With
/// `T = 20000` and the parameters below the expected count is around 20000, so a
/// Poisson standard error would be about 0.7%; the clustering factor at
/// `alpha = 0.5` multiplies that by 2. 5% is a few standard errors, loose enough not
/// to be flaky and tight enough that a wrong branching-ratio convention
/// (`alpha` vs `alpha/beta`, a factor of `beta`) fails it comfortably.
const MEAN_INTENSITY_TOLERANCE: f64 = 0.05;

/// Kolmogorov-Smirnov statistic of `sample` against the unit-rate exponential CDF.
fn ks_statistic_against_unit_exponential(sample: &mut [f64]) -> f64 {
    sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sample.len() as f64;
    let mut deviation: f64 = 0.0;
    for (index, &value) in sample.iter().enumerate() {
        let theoretical = 1.0 - (-value).exp();
        let below = index as f64 / n;
        let above = (index as f64 + 1.0) / n;
        deviation = deviation.max((theoretical - below).abs());
        deviation = deviation.max((above - theoretical).abs());
    }
    deviation
}

/// The stationary mean intensity `mu / (1 - alpha)` [Laub2015, eq. 6].
///
/// This is the oracle that catches a branching-ratio convention error. Under the
/// other convention in the literature the limit would be `mu / (1 - alpha/beta)`,
/// which differs by a factor of `beta` in the denominator — `beta = 2.0` below, so
/// the two predictions are far apart.
#[test]
fn converges_to_the_stationary_mean_intensity() {
    let parameters = Parameters::new(0.6, 0.5, 2.0).unwrap();
    let expected = parameters.stationary_mean_intensity().unwrap();
    assert!(
        (expected - 1.2).abs() < 1e-12,
        "mu/(1-alpha) = 0.6/0.5 = 1.2"
    );

    let horizon = 20_000.0;
    for seed in SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let times = simulate(&parameters, horizon, &mut rng).unwrap();
        let observed = times.len() as f64 / horizon;
        let relative = (observed - expected).abs() / expected;
        assert!(
            relative <= MEAN_INTENSITY_TOLERANCE,
            "seed {seed}: observed mean intensity {observed:?} vs analytic \
             {expected:?}, relative error {relative:?} > {MEAN_INTENSITY_TOLERANCE}. \
             Under the alpha/beta branching-ratio convention the prediction would be \
             {:?}.",
            parameters.baseline() / (1.0 - parameters.excitation() / parameters.decay())
        );
    }
}

/// Time-rescaling residuals must be `Exp(1)` [Laub2015, Theorem 4].
///
/// The KS critical value at the 1% level is `1.628 / sqrt(n)`. 1% rather than 5%
/// because this runs over several seeds and a 5% level would fail one in twenty by
/// construction; with 6 seeds at 1% the chance of a spurious failure is about 6%,
/// which is still visible but tolerable for a test that must not be flaky. The point
/// of the threshold is to catch a systematically wrong compensator, which produces a
/// statistic many times the critical value, not to do careful inference.
#[test]
fn time_rescaled_residuals_are_unit_exponential() {
    let cases = [
        (Parameters::new(0.8, 0.4, 1.5).unwrap(), 6000.0),
        (Parameters::new(1.2, 0.6, 3.0).unwrap(), 4000.0),
        (Parameters::new(0.5, 0.2, 0.7).unwrap(), 8000.0),
    ];

    for (parameters, horizon) in cases {
        for seed in SEEDS.iter().take(6) {
            let mut rng = ChaCha8Rng::seed_from_u64(*seed);
            let times = simulate(&parameters, horizon, &mut rng).unwrap();
            assert!(times.len() > 500, "too few events to test");

            let observation = Observation::new(&times, horizon).unwrap();
            let compensators = compensator_at_events(&parameters, &observation);

            // Successive differences of the transformed times. Lambda(0) = 0, so the
            // first residual is the first transformed time itself.
            let mut residuals = Vec::with_capacity(compensators.len());
            let mut previous = 0.0;
            for &value in &compensators {
                residuals.push(value - previous);
                previous = value;
            }

            let n = residuals.len() as f64;
            let statistic = ks_statistic_against_unit_exponential(&mut residuals);
            let critical = 1.628 / n.sqrt();
            assert!(
                statistic <= critical,
                "mu={} alpha={} beta={} seed={seed}: KS statistic {statistic:?} > \
                 critical {critical:?} at n={n}. Residuals are not unit exponential, \
                 so the simulator and the compensator disagree about the intensity.",
                parameters.baseline(),
                parameters.excitation(),
                parameters.decay(),
            );
        }
    }
}

/// The KS test must be able to reject. An oracle that accepts everything is not an
/// oracle, and this one is a statistic against a threshold rather than an exact
/// comparison, so its power is worth demonstrating rather than assuming.
#[test]
fn the_ks_test_rejects_residuals_from_the_wrong_parameters() {
    let truth = Parameters::new(0.8, 0.4, 1.5).unwrap();
    let horizon = 6000.0;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let times = simulate(&truth, horizon, &mut rng).unwrap();
    let observation = Observation::new(&times, horizon).unwrap();

    // Rescale with a deliberately wrong branching ratio.
    let wrong = Parameters::new(0.8, 0.1, 1.5).unwrap();
    let compensators = compensator_at_events(&wrong, &observation);
    let mut residuals = Vec::with_capacity(compensators.len());
    let mut previous = 0.0;
    for &value in &compensators {
        residuals.push(value - previous);
        previous = value;
    }
    let n = residuals.len() as f64;
    let statistic = ks_statistic_against_unit_exponential(&mut residuals);
    let critical = 1.628 / n.sqrt();
    assert!(
        statistic > critical,
        "KS statistic {statistic:?} did not exceed {critical:?} for residuals \
         computed with the wrong parameters; the test has no power"
    );
}

#[test]
fn simulated_realizations_satisfy_the_input_contract() {
    let parameters = Parameters::new(0.9, 0.5, 1.4).unwrap();
    for seed in SEEDS {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let times = simulate(&parameters, 500.0, &mut rng).unwrap();
        // Ascending, inside the window: Observation::new enforces both.
        Observation::new(&times, 500.0).expect("simulator must produce valid input");
        // Ogata thinning draws continuous inter-arrival times, so ties have
        // probability zero. This is why the tied fixtures had to be hand-built.
        assert!(
            times.windows(2).all(|w| w[0] < w[1]),
            "seed {seed}: simulator produced a tie, which should have probability zero"
        );
    }
}

#[test]
fn simulation_is_reproducible_from_a_seed() {
    let parameters = Parameters::new(0.7, 0.45, 1.1).unwrap();
    let first = simulate(&parameters, 300.0, &mut ChaCha8Rng::seed_from_u64(99)).unwrap();
    let second = simulate(&parameters, 300.0, &mut ChaCha8Rng::seed_from_u64(99)).unwrap();
    assert_eq!(first, second, "same seed must give the same realization");
}
