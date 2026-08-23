# hawk

Multivariate Hawkes processes in Rust: simulation and maximum-likelihood estimation,
with Python bindings.

> ## Pre-alpha
>
> The exponential-kernel process works in one and `d` dimensions: simulation,
> log-likelihood, analytic gradient and maximum-likelihood fitting. **There are no
> Python bindings yet** and the API will change. Do not depend on this crate.
>
> Every formula traces to an approved derivation, and every oracle has been shown to
> go red on deliberately broken code — see
> [`docs/verification-log.md`](docs/verification-log.md).

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
| M1 | Univariate exponential-kernel likelihood, gradient, simulator, MLE | complete |
| M2 | Multivariate with cross-excitation | complete |
| M3 | Python wheels | not started |
| M4 | Benchmarks against `tick` | partial — see `docs/positioning-probe.md` |

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

- **Brute-force reference** — the `O(n^2)` definition, validated against hand
  calculations and the Poisson closed form. The `O(n)` recursion is gated against it.
- **Differential test against `tick`** — ten committed scenarios, four parameter
  points each.
- **Round-trip property test** — simulate, fit, recover, with the tolerance derived
  per realization from the observed Fisher information.
- **Finite-difference gradient check** — a wrong derivative still converges, to the
  wrong place, and nothing else catches it. `tick` cannot check `d/dbeta` at all.
- **Analytic identity** — the stationary mean intensity `mu/(1-alpha)`.
- **Time-rescaling residuals** — KS-tested against `Exp(1)`. Validates the simulator
  and the compensator jointly, and catches compensator bugs the other oracles miss.

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
