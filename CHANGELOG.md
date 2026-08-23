# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — M1, univariate exponential-kernel Hawkes

First public API. `hawk::univariate`:

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

### Changed

- `negative_log_likelihood` computes the value in one pass without computing the
  gradient. It previously delegated to `negative_log_likelihood_and_gradient` and
  discarded the gradient. The returned value is bitwise unchanged, enforced by
  `hawk/tests/bit_identical_evaluation.rs`; fitted parameters are identical digit for
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
- Differential test against `tick` now runs `hawk` rather than a stub.
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
  9% on a four-event example. `hawk` groups by distinct time instead. [Laub2015]
  derives eq. 20 for a *simple* point process, so this is a hypothesis that does not
  survive `hawk`'s input contract rather than an error in the paper.
- On tied data the objective is not a likelihood, so the MLE asymptotics do not apply.
  The arithmetic is unaffected. See `univariate_loglikelihood.md` §3.1.
- Multivariate remains unimplemented; the multivariate fixtures are parsed and
  structurally validated but not yet compared.

### Added — M0, verification infrastructure

- Cargo workspace: `hawk` (core) and `hawk-python` (bindings placeholder).
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
  ordering and exact ties) are OPEN and must be closed before or during M1.
- OQ-10 records that CLAUDE.md's premise — that `tick` is unmaintained and breaks on
  Python 3.13+ — was not borne out, and needs a decision from the repository owner.
