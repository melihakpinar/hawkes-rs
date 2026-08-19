# hawk

Multivariate Hawkes processes in Rust: simulation and maximum-likelihood estimation,
with Python bindings.

> ## Pre-alpha — not yet usable
>
> **`hawk` implements no algorithms yet.** There is no simulator, no likelihood and
> no estimator. Nothing here will fit a model to your data. Do not depend on this
> crate.
>
> What exists today is the verification machinery that the algorithms will have to
> satisfy, built first on purpose. See [`docs/verification-log.md`](docs/verification-log.md)
> for evidence that it detects the failures it is meant to detect.

## Why

The library's only real product is **correct numbers**. Speed is secondary. A fast
library that returns subtly wrong parameter estimates is worse than nothing, because
users cannot tell.

That priority is why the verification harnesses landed before any algorithm, and why
every formula in this repository has to trace to a citable source before it may be
written. The rules are in [`CLAUDE.md`](CLAUDE.md).

## Status

| Milestone | Contents | State |
| --- | --- | --- |
| M0 | Verification infrastructure: pinned `tick` oracle, reference fixtures, differential / property / gradient harnesses, sabotage evidence | complete |
| M1 | Univariate exponential-kernel likelihood, gradient, simulator, MLE | not started |
| M2 | Multivariate, Python wheels, benchmarks against `tick` | not started |

v0.1.0 is univariate and multivariate exponential-kernel Hawkes: simulation, MLE,
Python wheels, benchmarks. Sum-of-exponentials and power-law kernels, non-parametric
estimation, regularization, marked and spatial processes are explicitly out of scope.

## Layout

```
hawk/               Rust core crate (no algorithms yet)
hawk-python/        PyO3 bindings (placeholder until M1)
docs/
  derivations/      approved derivations; conventions.md pins the index conventions
  references/       papers (not committed)
  open-questions.md unresolved conventions and BLOCKED work
  verification-log.md  proof that each oracle goes red when the code is broken
benchmarks/docker/  pinned tick environment and the fixture generator
tests/fixtures/     committed reference data from tick
```

## Verification

Three of CLAUDE.md's five oracles are live. Each has been observed failing on
deliberately broken code before being trusted:

- **Differential test against `tick`** — six committed scenarios, four parameter
  points each, agreeing to `1e-9`.
- **Round-trip property test** — `proptest` over stationary parameters.
- **Finite-difference gradient check** — central differences against closed-form
  gradients; a wrong derivative still converges, to the wrong place, and nothing else
  catches it.

The analytic-identity and time-rescaling oracles arrive with the simulator in M1.

### Running

```sh
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

Regenerating the fixtures needs Docker; see
[`benchmarks/docker/README.md`](benchmarks/docker/README.md). It is not required to
run the test suite — the fixtures are committed.

## A note on `tick`

`tick` is the differential-test oracle. Working with it turned up several things
worth knowing before you reach for it yourself, including an undeclared runtime
dependency without which every model class fails to construct, and the fact that
`ModelHawkesExpKernLogLik.loss` is neither the negative log-likelihood nor the
formula in its own docstring. Both are written up in
[`docs/open-questions.md`](docs/open-questions.md) (OQ-9, OQ-8) and
[`docs/derivations/conventions.md`](docs/derivations/conventions.md).

## Licence

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your
option.
