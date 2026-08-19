# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet. `hawk` exposes no public API: M0 ships verification infrastructure only.

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
