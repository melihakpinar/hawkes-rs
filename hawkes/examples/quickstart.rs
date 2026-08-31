//! The example in the README. Kept here so CI compiles and runs it.
//!
//! Simulates a univariate exponential-kernel Hawkes process, fits it back, and reports
//! the stationarity diagnostic.

use hawkes::univariate::{Observation, Parameters, fit, simulate};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), hawkes::Error> {
    let truth = Parameters::new(0.5, 0.6, 1.0)?;

    // The horizon is supplied, never inferred from the events. An inferred window
    // silently biases the baseline; see docs/derivations/conventions.md C5.
    let horizon = 20_000.0;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let times = simulate(&truth, horizon, &mut rng)?;

    let observation = Observation::new(&times, horizon)?;
    let result = fit(&observation)?;
    let fitted = &result.parameters;

    println!("{} events on [0, {horizon}]", times.len());
    println!("  baseline   {:.4}  (true 0.5)", fitted.baseline());
    println!("  excitation {:.4}  (true 0.6)", fitted.excitation());
    println!("  decay      {:.4}  (true 1.0)", fitted.decay());
    println!(
        "  converged={} iterations={}",
        result.converged, result.iterations
    );

    // Stationarity is a diagnostic on the result, not a constraint during fitting:
    // a non-stationary fit is a real finding about the data (CLAUDE.md §6).
    println!(
        "  branching ratio {:.4} -> stationary={}",
        fitted.branching_ratio(),
        fitted.is_stationary()
    );
    Ok(())
}
