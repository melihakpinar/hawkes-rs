# hawkes-rs

[![CI](https://github.com/melihakpinar/hawkes-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/melihakpinar/hawkes-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hawkes-rs.svg)](https://crates.io/crates/hawkes-rs)
[![PyPI](https://img.shields.io/pypi/v/hawkes-rs.svg)](https://pypi.org/project/hawkes-rs/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#licence)

Multivariate Hawkes processes in Rust, with Python bindings: simulation by Ogata
thinning, the exponential-kernel log-likelihood with its analytic gradient in one `O(n·d)`
pass, and maximum-likelihood estimation of the baseline, the excitation matrix **and the
decay**. Every formula traces to an approved derivation in
[`docs/derivations/`](docs/derivations/), and every test oracle has been deliberately
broken and watched go red before being trusted. It is slower than `tick` on the
univariate fit; the numbers and the reasons are below.

## Install

```sh
pip install hawkes-rs                          # Python 3.9+, numpy >= 1.22
cargo add hawkes-rs rand@0.9 rand_chacha@0.9   # Rust 1.85+
```

`simulate` takes an `impl rand::Rng`, so a Rust caller's `rand` must be the 0.9 series
this crate is built against; a bare `cargo add rand` resolves to 0.10 and fails on the
trait bound. That coupling is tracked as
[#42](https://github.com/melihakpinar/hawkes-rs/issues/42).

## Quickstart

Both programs below are committed and run by CI, which also checks that these blocks
are the files: [`quickstart.py`](hawkes-python/examples/quickstart.py),
[`quickstart.rs`](hawkes/examples/quickstart.rs). A three-component example with
residual analysis is [`fit_multivariate.py`](hawkes-python/examples/fit_multivariate.py).

```python
from __future__ import annotations

import numpy as np

from hawkes import univariate


def main() -> None:
    truth = univariate.Parameters(baseline=0.5, excitation=0.6, decay=1.0)

    # `horizon` is supplied, never inferred from the events. An inferred window
    # silently biases the baseline; see docs/derivations/conventions.md C5.
    horizon = 20_000.0
    times = univariate.simulate(truth, horizon, seed=7)

    fit = univariate.fit(times, horizon)
    print(f"{len(times)} events on [0, {horizon:g}]")
    print(f"  baseline   {fit.parameters.baseline:.4f}  (true {truth.baseline})")
    print(f"  excitation {fit.parameters.excitation:.4f}  (true {truth.excitation})")
    print(f"  decay      {fit.parameters.decay:.4f}  (true {truth.decay})")
    print(f"  converged={fit.converged} iterations={fit.iterations}")

    # Stationarity is a diagnostic on the result, not a constraint during fitting:
    # a non-stationary fit is a real finding about the data (CLAUDE.md §6).
    ratio = fit.branching_ratio()
    print(f"  branching ratio {ratio:.4f} -> stationary={fit.is_stationary()}")


if __name__ == "__main__":
    main()
```

```rust
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
```

Both print 24 899 events and a fitted baseline, excitation and decay of 0.5041, 0.5951
and 0.9852 against true values of 0.5, 0.6 and 1.0.

## What it guarantees, and what it does not

- **The kernel is `alpha * beta * exp(-beta * t)`**, `tick`'s parametrization, so
  `alpha` is the branching ratio directly and `excitation[i][j]` means "j excites i"
  ([`conventions.md`](docs/derivations/conventions.md) C1, C2, C6).
- **The observation window `[0, T]` is caller-supplied** and never inferred from the
  data. Trailing dead time lowers the fitted baseline, as it should (C5).
- **The decay is estimated**, together with the baseline and the excitation. One decay
  is shared by every pair; per-pair decays are out of scope for 0.1.
- **Ties are pooled by distinct time**, so simultaneous events, within or across
  components, do not excite each other, and unsorted input is rejected rather than
  sorted (C3, C8). On tied data the objective is still the right number but is not a
  likelihood, so the usual asymptotics do not apply.
- **Stationarity is diagnosed, not enforced.** The fit reports the branching ratio, or
  the spectral radius of the excitation matrix, and leaves the verdict to you. A
  non-stationary fit is a finding about the data.
- **Positivity is by parametrization**: the optimizer runs in log space, so a true zero
  in the excitation matrix comes back as a small positive number
  ([`parameter_space.md`](docs/derivations/parameter_space.md)).
- **No regularization, no other kernels, no marks.** Univariate and multivariate
  exponential kernels, simulation and MLE. That is all of 0.1.

Correctness is checked five ways: the `O(n)` recursion against a brute-force
transcription of the definition; the analytic gradient against central differences;
the stationary mean intensity and time-rescaled residuals of the simulator; a
simulate-then-fit round trip with tolerances from the Fisher information; and a
differential test against `tick` on committed fixtures.
[`docs/verification-log.md`](docs/verification-log.md) records each oracle going red
under sabotage.

## Compared with tick

`tick` is faster on the univariate fit, by about 3x at a million events. Same events,
same machine, single-threaded, median of 5 runs. Three asymmetries could not be
equalised and all favour `tick`: it is handed the true decay rather than estimating it,
its objective carries an L2 penalty, and its stopping criterion is not the same kind of
criterion. Every ratio is **hawkes-rs / tick**; above 1 means `tick` is faster.

| events | hawkes-rs | tick, likelihood | hawkes-rs / tick | tick, least-squares | hawkes-rs / tick |
| --- | --- | --- | --- | --- | --- |
| 978 | 0.0021 | 0.0010 | 2.14x | 0.0005 | 4.39x |
| 10,122 | 0.0117 | 0.0037 | 3.19x | 0.0008 | 13.83x |
| 99,718 | 0.0946 | 0.1248 | 0.76x | 0.0050 | 18.97x |
| 1,000,453 | 0.9509 | 0.2987 | 3.18x | 0.0464 | 20.49x |

Seconds, from [`fit-d1.json`](benchmarks/results/fit-d1.json). What `hawkes-rs` does
that `tick` does not, each measured in [`benchmarks/results/`](benchmarks/results/):
`tick`'s learner cannot express an observation window, so on one realization with a true
baseline of 0.5 it returns 0.5133 whether the declared window has 0% or 50% trailing
dead time, while `hawkes-rs` moves from 0.5000 to 0.0004
([`window-bias.json`](benchmarks/results/window-bias.json)); it cannot estimate the
decay; its documented likelihood interface raises at `d > 1`, so at `d = 10` and
`d = 100` the comparison is against its least-squares default, a different estimator;
and its `loss` is neither the log-likelihood nor its own docstring's formula. At
`d = 100` `hawkes-rs` does not converge within its iteration cap; that is a limitation,
stated with the diagnosis in [`benchmarks/README.md`](benchmarks/README.md), along with
the methodology, the `d = 10`, `d = 100` and simulation tables, and where else `tick`
wins.

## Documentation

- [`docs/derivations/`](docs/derivations/) — every formula, with its conventions and
  citations; `conventions.md` pins each CLAUDE.md §1.3 hazard to a source.
- [`docs/open-questions.md`](docs/open-questions.md) — OQ-1 to OQ-11, all resolved.
- [`docs/verification-log.md`](docs/verification-log.md) — each oracle shown to go red.
- [`benchmarks/README.md`](benchmarks/README.md) — methodology, fixed before any number.
- [`CHANGELOG.md`](CHANGELOG.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
  [`CLAUDE.md`](CLAUDE.md) — the rules this repository is maintained under.

## Supported versions

Rust 1.85 or later, the crate's `rust-version`. Python 3.9 or later on `abi3` wheels
for Linux `x86_64` and `aarch64`, macOS `arm64` and Windows `AMD64`; the macOS
`x86_64` wheel is built but has not been loaded by an interpreter
([`docs/python-wheels.md`](docs/python-wheels.md)).

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
