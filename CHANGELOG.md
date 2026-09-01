# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The benchmark and probe binaries moved from `hawkes/examples/` to the workspace
  member `benchmarks/tooling/` (`cargo build --release -p hawkes-benchmarks`); the
  suite scripts follow. `hawkes/examples/` now holds `quickstart.rs`, the example a
  user runs, and `dump_fixture_nll.rs`, which the Python bit-identity test depends on.
  No library code path changed: every fixture negative log-likelihood bit pattern,
  the quickstart output and seeded simulations are identical before and after.

## [0.1.2] — 2026-09-01

README corrections and two fixes to the publish workflow. No library code changed.

### Changed

- README: the Rust install line is `cargo add hawkes-rs` alone. The `rand@0.9` pin and its
  explanation moved below the quickstart, where the example needing them is visible, and
  now say what the coupling is — `simulate` takes an `impl rand::Rng`, so `rand` is in the
  public signature and every future `rand` major is a breaking change for callers (#42,
  for v0.2).
- README: every ratio reads **hawkes-rs / tick** and both table headers say so. The
  simulation table had been inverted, so `0.54x` and `3.18x` pointed opposite ways on one
  page. `benchmarks/suite/readme_tables.py` matches, so regeneration reproduces the README
  rather than reverting it.

### Fixed

- The publish job now checks that the tag names the version the manifests build, before
  anything is downloaded or uploaded. `v0.1.2` on a commit whose manifests still said
  `0.1.1` rebuilt 0.1.1, uploaded nothing, and failed two minutes later as a sha mismatch —
  a symptom three steps from the cause. An optional `-suffix` is stripped, so a deliberate
  re-publish tag like `v0.1.0-publish` still matches.
- `verify_pypi_release.py` compares sha256 only for the files the run actually uploaded,
  decided by snapshotting the index before the upload. Wheels are not bit-reproducible:
  rebuilding 0.1.1 gave five wheels differing from the published ones by one to three
  bytes, while the sdist was byte-identical. Presence is still required for every file.

## [0.1.1] — 2026-08-31

**Metadata only. No code changed.** `hawkes/src` and `hawkes-python/src` are byte for byte
identical to 0.1.0; this release exists to correct what the registries display.

crates.io and PyPI embed the README as it stood at publish time, and 0.1.0 was published
before the post-release README edits. Both pages therefore still tell a reader that
"Neither package is published yet" and hand them `cargo add --git …` and a wheel built from
a checkout — their own installation instructions are wrong, and no amount of editing the
repository fixes a page that was snapshotted at upload.

- README rewritten: install instructions that work, the hook stated as the observation
  window rather than a speed claim, and the speed losses named in the second paragraph.
- Version bumped in `Cargo.toml` and `hawkes-python/pyproject.toml`.

Published from CI. This is also the first release where
`.github/scripts/verify_pypi_release.py` runs against a real upload: all six files are new
at 0.1.1, so nothing is skipped and the check has something to confirm rather than a
release that was already complete before it ran.

## [0.1.0] — 2026-08-31

First release. Univariate and multivariate exponential-kernel Hawkes processes:
simulation, log-likelihood, analytic gradient, maximum-likelihood estimation, Python
wheels, and a benchmark suite measured against `tick`.

### Published, and what installing it showed

`hawkes-rs` 0.1.0 is on crates.io and PyPI. The README's install block now gives the
real commands, verified against the published artifacts rather than the locally built
ones: `pip install hawkes-rs` in a clean virtualenv in a directory with no source tree,
and `cargo add hawkes-rs` in an empty crate, both reproducing the quickstart figures
digit for digit — 24 899 events, baseline 0.5041, excitation 0.5951, decay 0.9852.

Two things that only the published artifacts could show:

- **The Rust quickstart does not compile with a bare `cargo add rand`.** `simulate`
  takes an `impl rand::Rng`, so the caller's `rand` must be the same major version the
  crate was built against. `cargo add rand` resolves to 0.10 and fails on the trait
  bound; `rand@0.9` and `rand_chacha@0.9` are required, and the README now says so.
  Leaking a dependency's types through the public API is what makes this the caller's
  problem, and it is worth revisiting before the API stabilises.
- **PyPI carries only the macOS `arm64` wheel and an sdist.** The release was published
  from a laptop, so the Linux, Windows and Intel-macOS wheels the CI matrix builds were
  never uploaded, and `pip install` on those platforms falls back to compiling the sdist
  and needs a Rust toolchain. Publishing from CI would close this.

### Added — M4, benchmarks and the README

- `benchmarks/README.md`: methodology fixed and committed before any number was
  produced — warmup, repetitions, statistic, grid, threads, hardware, and the
  asymmetries that could not be equalised.
- `benchmarks/suite/`: `fit_d1`, `fit_d10`, `fit_d100`, `simulate` and `window_bias`,
  each standalone, each writing committed JSON to `benchmarks/results/`.
- Every benchmark records both libraries' fitted parameters and scores them under
  `hawkes`'s objective, so two different objectives sit in one unit.
- `benchmarks/suite/create_diagrams.py` regenerates every chart from the committed JSON
  with no plotting dependency and no manual step.
- `benchmarks/suite/readme_tables.py` regenerates the README's benchmark tables from the
  same JSON, so every published number is re-checkable by diff.
- `hawkes/examples/quickstart.rs` and `hawkes-python/examples/quickstart.py`: the README's
  usage examples, compiled, run and type-checked by CI so they cannot drift from the
  prose.
- README rewritten around what was measured, including where `tick` wins.

### Measured

- `tick` cannot fit a multivariate likelihood through `HawkesExpKern`: every
  deterministic solver leaves the non-negative region its C++ model requires and raises.
  `sgd` returns and is visibly wrong. At `d > 1` the comparison is therefore against
  `tick`'s least-squares default, a different estimator.
- `tick`'s learner takes no observation window, so its baseline cannot respond to one.
  `benchmarks/results/window-bias.json` records the consequence and the
  `HawkesExpKern.fit` signature alongside it.


### Added — M1, univariate exponential-kernel Hawkes

First public API. `hawkes::univariate`:

- `Parameters` — validated `mu`, `alpha`, `beta`, with `branching_ratio`,
  `is_stationary` and `stationary_mean_intensity` [Laub2015, eq. 5, 6].
- `Observation` — a realization on `[0, T]`, validated against the input contract
  (`conventions.md` C8). `T` is supplied by the caller and never inferred.
- `negative_log_likelihood`, `negative_log_likelihood_and_gradient` — `O(n)`, via the
  grouped recursion, with the analytic gradient in the same pass.
- `Gradient::to_log_parameter_space` — the chain-rule conversion (G.8).
- `compensator_at_events` — for Ogata residual analysis.
- `simulate` — Ogata thinning [Laub2015, Algorithm 2].
- `fit` — maximum likelihood by L-BFGS in log-parameter space, with the branching
  ratio reported as a diagnostic rather than enforced.
- `Fit::objective_evaluations` and `Fit::gradient_evaluations` — evaluation counts for
  a fit, counting line-search trials rather than iterations.

### Added — M2, multivariate

`hawkes::multivariate`, `d` components with cross-excitation and a shared decay:

- `Parameters` — baseline vector and row-major excitation matrix, `alpha[i][j]` meaning
  "j excites i". `branching_ratio_spectral_radius`, `is_stationary`,
  `stationary_mean_intensity` = `(I - alpha)^-1 mu` [Bacry2015, Prop. 4 eq. 21].
- `Observation`, `negative_log_likelihood`, `negative_log_likelihood_and_gradient`,
  `compensator_at_events`, `simulate`, `fit`, `fit_from`.
- `negative_log_likelihood_parallel` behind the off-by-default `rayon` feature.

At `d = 1` the multivariate path is **bitwise identical** to `univariate`, for both
value and gradient.

### Fixed

- Fixture schema 3 drops `spectral_radius`. It came from LAPACK, whose kernel selection
  depends on the CPU it runs on, which made the `d = 10` fixture pass and fail across CI
  runs of identical content. It is derivable from `adjacency`, which is in the file.

- `multivariate::Parameters::branching_ratio_spectral_radius` returned `1.0` for a
  nilpotent excitation matrix, whose spectral radius is `0`, so a cascade that dies out
  after finitely many steps was reported as explosive and had no stationary mean
  intensity. Caused by an early exit that fired while the upper bound was temporarily
  plateaued. Found by widening the regression cases to defective matrices.

### Notes on M2

- The recursion groups distinct times **pooled across all components**. Advancing per
  event lets whichever component a merge visits first excite the others at a shared
  timestamp; on a four-event example that is a 1.1% error, and it is invisible on any
  simulated data.
- `fit` optimizes `alpha` in log space, so a fitted entry is never exactly zero. The
  reasoning and what it costs are in `docs/derivations/parameter_space.md`.
- `stationary_mean_intensity` checks the spectral radius rather than inferring
  stationarity from the linear solve succeeding; `I - alpha` can be invertible with
  spectral radius above 1, and the solve then returns a vector with negative entries.
- The `rayon` path is bitwise identical to the sequential one and **228x to 397x
  slower** (`benchmarks/results/multivariate-parallel.json`). It is off by default.
- A `d = 10` fixture was added, giving the corpus a second value of `D` so the `D*T`
  offset in the OQ-8 identity is pinned rather than coincidental.

### Changed

- **Breaking.** `multivariate::negative_log_likelihood`,
  `negative_log_likelihood_and_gradient`, `compensator_at_events` and
  `negative_log_likelihood_parallel` return `Result` instead of the bare value. They
  previously panicked when the `Parameters` and the `Observation` disagreed on the
  number of components. Both halves come from the caller, so that is invalid input and
  CLAUDE.md §5 makes it an error value. Callers add `?` or `.unwrap()`; nothing about
  the computed numbers changed.
- `Error::ProcessDimensionMismatch` is the new variant, distinct from
  `DimensionMismatch`, which remains about the internal shape of a single `Parameters`
  (`baseline` against `excitation`).
- `hawkes-python` no longer carries a `check_dimensions` shim. The `ValueError` a Python
  caller sees is now raised from the same check a Rust caller gets, so the two cannot
  drift apart.
- `negative_log_likelihood` computes the value in one pass without computing the
  gradient. It previously delegated to `negative_log_likelihood_and_gradient` and
  discarded the gradient. The returned value is bitwise unchanged, enforced by
  `hawkes/tests/bit_identical_evaluation.rs`; fitted parameters are identical digit for
  digit. End-to-end fit time at `n = 1e6` went from 1.003165 s to 0.853488 s
  (`docs/positioning-probe.md` §20).

### Verification

- Brute-force `O(n^2)` reference, validated against hand calculations and the Poisson
  closed form. The `O(n)` recursion is gated against it to `1e-12`, relative to the
  computation scale rather than to `|nll|`.
- Analytic gradient against central differences to `1e-6`, in both natural and
  log-parameter space.
- Stationary mean intensity, and time-rescaled residuals KS-tested against `Exp(1)`,
  with a permanent negative control.
- Round-trip property test over 200 cases, tolerance derived per realization from the
  observed Fisher information.
- Differential test against `tick` now runs `hawkes` rather than a stub.
- Thirteen further sabotages recorded in `docs/verification-log.md` (S10-S22).

### Resolved

- **OQ-8** — `tick`'s loss is the negative log-likelihood ratio against a unit-rate
  Poisson, divided by `n_jumps`. The identity
  `hawk_nll == tick_loss * n_jumps + D*T` holds with excitation present and with
  ties. Absolute comparison against `tick` is now trustworthy, given the conversion.
- **OQ-7** — input contract: ascending within a component, ties permitted and
  non-mutually-exciting, `[0, T]` inclusive.
- **OQ-11** — [Laub2015] is in `docs/references/`; equations cited by number.

### Notes

- The textbook Ozaki recursion [Laub2015, eq. 20] is **wrong** on tied input, by about
  9% on a four-event example. `hawkes` groups by distinct time instead. [Laub2015]
  derives eq. 20 for a *simple* point process, so this is a hypothesis that does not
  survive `hawkes`'s input contract rather than an error in the paper.
- On tied data the objective is not a likelihood, so the MLE asymptotics do not apply.
  The arithmetic is unaffected. See `univariate_loglikelihood.md` §3.1.
- Multivariate remains unimplemented; the multivariate fixtures are parsed and
  structurally validated but not yet compared.

### Added — M0, verification infrastructure

- Cargo workspace: `hawkes` (core) and `hawkes-python` (bindings placeholder).
- Pinned `tick` oracle image under `benchmarks/docker/`: CPython 3.13.5,
  `tick` 0.8.0.2, `linux/amd64`, complete pinned dependency closure.
- Six reference fixtures in `tests/fixtures/` — univariate through trivariate,
  13 to 3100 events, four parameter points each — reproducible byte-for-byte.
- Differential-test harness against `tick`, agreeing to `1e-9`.
- `proptest` round-trip harness over stationary parameters.
- Central-difference gradient harness, validated against closed-form gradients.
- `docs/verification-log.md`: ten sabotages, each confirmed to turn a harness red.
- `docs/derivations/conventions.md`: the CLAUDE.md §1.3 convention hazards pinned to
  `tick`'s source.
- `docs/open-questions.md`: OQ-1 through OQ-10.
- CI running `fmt`, `clippy -D warnings` and `test`.

### Notes

- No algorithm code. No intensity, likelihood, simulator or estimator exists.
- OQ-8 (whether `tick`'s loss offset is parameter-independent) and OQ-7 (event
  ordering and exact ties) were OPEN at the time of this milestone; both were closed
  in M1.
- OQ-10 recorded that CLAUDE.md's premise — that `tick` is unmaintained and breaks on
  Python 3.13+ — was not borne out. Closed after the positioning probe measured it;
  the preamble no longer makes that claim.
