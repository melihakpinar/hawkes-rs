# Benchmark methodology

**This document was written and committed before any timing number was produced.**
Results live in `benchmarks/results/` as JSON and are referenced from the README; the
numbers were appended to this repository after this file was fixed, and nothing here
was changed once they existed. That ordering is the point: choosing a methodology after
seeing the numbers is how benchmarks become dishonest with nobody deciding to be
dishonest.

`docs/positioning-probe.md` is the univariate precedent and this file follows its
structure deliberately.

## 0. Speed is not this library's claim

The positioning probe measured `tick` at roughly **3x faster** than `hawk` on the
univariate fit at `n = 1e6`. That result stands, is expected to reproduce here, and is
reported in the README in the same typeface as everything else. `hawk` exists for
reasons that are not speed, and they are listed in `CLAUDE.md`.

A benchmark section with no losses in it is marketing. The cases where the incumbent
wins are reported first.

## 1. What is measured

Four benchmarks, one runnable script each:

| Script | What it times |
| --- | --- |
| `suite/fit_d1.*` | one univariate maximum-likelihood fit |
| `suite/fit_d10.*` | one 10-component fit |
| `suite/fit_d100.*` | one 100-component fit |
| `suite/simulate.*` | one simulation to a fixed horizon |

Not measured: memory, allocation behaviour, startup cost, data loading, model
construction that can be hoisted, or accuracy as a statistical property of the
estimators.

## 2. Environment

| | |
| --- | --- |
| Machine | Apple M2, 8 cores, 16 GiB |
| Architecture | `arm64`, native on both sides |
| OS | macOS, Darwin 22.6.0 |
| `hawk` | this working tree, `cargo build --release` |
| Python | 3.14.5 |
| `tick` | 0.8.0.2 |
| `numpy` / `scipy` | 2.5.2 / 1.18.1 |
| Threads | 1 on both sides |

`hawk` is single-threaded and has no parallelism on the paths measured here. The `tick`
process runs with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` all set to 1; `HawkesExpKern` exposes
no `n_threads` argument, so the environment is the only control available.

Both sides run natively. The pinned `linux/amd64` image in `benchmarks/docker/` runs
under emulation on this host and remains the correctness oracle only — emulated timings
compared against native ones would measure the emulator.

## 3. Data

Generated once per configuration by **`hawk`'s** simulator, seeded, written to disk in
Rust's shortest-round-trip decimal form, and read back by both sides, so both fit
exactly the same events. Generation is outside every clock.

| | |
| --- | --- |
| Seed | 20260831, fixed |
| Decay | `beta = 1.0` |
| Baseline | `mu_i = 0.5 / d` |
| Excitation | `alpha_ij = 0.6 / d` |
| Spectral radius | 0.6 at every `d`, so every configuration is stationary and comparably far from the boundary |
| Horizon | `T = n / (sum_i m_i)`, with `m` the stationary mean intensity, so the expected total count is the nominal `n` |

Holding the spectral radius at 0.6 across `d` is the choice that makes the dimensions
comparable. Holding `alpha_ij` fixed instead would make the `d = 100` process explosive.

The realized event count differs from the nominal `n` because simulation is random; the
realized count is reported everywhere.

For the `simulate` benchmark the seed is advanced per repetition, so no repetition
reuses another's random stream and the timing is not that of one lucky realization. The
two libraries use different generators, so their realized counts differ from each other
even at the same horizon; both counts are reported and the comparison is stated per
event.

## 4. Timing protocol

- 1 warmup run, discarded.
- 5 timed runs.
- **Median** of the 5 reported, with `[min, max]` of the same 5 so dispersion is visible.
- `std::time::Instant` on the `hawk` side, `time.perf_counter()` on the `tick` side.
  Both are monotonic wall clocks.
- The clock covers the fit or simulate call only.

### 4.1 The budget rule

Fixed here, in advance, and applied identically to both libraries:

- A single timed run that exceeds **600 s** aborts its cell.
- A cell — one library, one configuration, warmup plus all 5 runs — that exceeds
  **1800 s** aborts.

An aborted cell is reported as **not completed within budget**, with the budget stated.
It is not retried with different settings, and its absence is never presented as a win
for the other side.

This rule exists because the capability probe of §5.4 showed `d = 100` to be slow enough
on `tick`'s side that an unbounded run was not practical. The rule was written before
any figure was recorded, and it cannot favour either library because it is the same
number for both.

## 5. What could not be equalized

Recorded rather than adjusted away, because adjusting them would mean measuring
something other than what each library does.

### 5.1 `tick` does not fit `beta`; `hawk` does

`HawkesExpKern(decays=...)` takes the decay as a fixed constructor argument. `tick` has
no exponential-kernel estimator that treats it as free.

- `hawk` solves a `d + d^2 + 1` parameter problem.
- `tick` solves a `d + d^2` problem with `beta` supplied.

`tick` is given the **true** `beta = 1.0`, the most favourable case for it: it is handed
for free the parameter the other side has to find. These are different optimization
problems and every time difference includes that.

### 5.2 `tick`'s objective carries an L2 penalty; `hawk`'s does not

`penalty="none"` is not reachable: with `solver="bfgs"` it raises
`ValueError: BFGS only accepts ProxZero and ProxL2sq for now`, and the other solvers
fail as in §5.4. `C = 1e9` is used; the probe established that `C = 1e9` and `C = 1e12`
agree to nine significant figures, so the penalty is not what determines the answer. It
is still not literally the same objective.

### 5.3 The stopping criteria are not the same kind of criterion

| | `hawk` | `tick` |
| --- | --- | --- |
| Solver | L-BFGS (`argmin` 0.11), More-Thuente line search, 7 correction pairs univariate / 10 multivariate | `HawkesExpKern`, `solver="bfgs"` |
| Space | log-parameter space, objective per event | natural parameters, penalized |
| Stop | gradient tolerance `1e-10` | `tol = 1e-10` |
| Iteration cap | 500 univariate, 1000 multivariate | 500 |

> **Correction.** This row first read "500" for `hawk` in both columns. That was wrong:
> `univariate::fit` caps at 500 and `multivariate::fit` at 1000
> (`hawk/src/univariate.rs:719`, `hawk/src/multivariate.rs:1056`). The error was found
> while reading the `d = 100` result, and is corrected here rather than quietly. It is a
> misdescription of existing code, not a setting changed after seeing numbers — no cap
> was altered, and no measurement moved.
>
> The correction-pair count in the row above was wrong the same way, and for the same
> reason: 7 is the univariate value (`hawk/src/univariate.rs:704`), 10 the multivariate
> one (`hawk/src/multivariate.rs:1049`). Both errors came from describing `hawk` from the
> univariate code alone.

`hawk` stops on a gradient norm; `tick`'s `tol` feeds its own solver's criterion, which
is not the same quantity. **Times measured under different convergence criteria are not
directly comparable.** Both are set as tight as their interfaces allow, and what each
actually achieved is reported.

### 5.4 `tick` cannot fit a multivariate likelihood through this interface

Established by interface probing before this methodology was fixed, on `d = 10` data
with 20 331 events. Every combination of `gofit="likelihood"` with a deterministic
solver fails:

| solver | penalty | outcome at `d = 10` |
| --- | --- | --- |
| `bfgs` | `l2` | `RuntimeError: The sum of the influence on someone cannot be negative` |
| `bfgs` | `none`, `l1`, `elasticnet` | `ValueError: BFGS only accepts ProxZero and ProxL2sq for now` |
| `agd`, `gd`, `svrg` | all four | `RuntimeError: The sum of the influence on someone cannot be negative` |
| `sgd` | all four | runs, and returns visibly wrong parameters (see below) |

The `RuntimeError` is the optimizer stepping outside the non-negative region that the
C++ model requires; nothing in the interface constrains it to stay there. `sgd` is the
only combination that returns, warns that its step size "needs to be tuned manually",
and at the true `alpha_ij = 0.06` returns `0.164` (`l2`), `0.097` (`none`), or `0.000`
with a baseline inflated to 1.69 and 7.39 (`elasticnet`, `l1`).

At `d = 1` the same likelihood path works with `bfgs` and fails with `agd`.

**Consequence for this benchmark.** At `d > 1` the only working `tick` estimator is
`gofit="least-squares"`, which is its default and a **different objective**: hawk
maximizes the likelihood, `tick` minimizes a least-squares contrast. The `d = 10` and
`d = 100` benchmarks therefore compare two different estimators, not two
implementations of one. This is stated wherever those numbers appear, and §5.5 is how
the difference is made visible rather than assumed.

At `d = 1`, where both objectives work, **both are measured**. That is what makes the
`d > 1` numbers interpretable: it shows what changing the objective costs on a case
where the change is the only difference. Without it, the least-squares timings at
`d = 10` and `d = 100` would be uninterpretable — fast, but with no way to tell how much
of the speed is the objective and how much is the implementation.

This is a capability finding, not a timing one, and it is reported as such.

### 5.5 Putting both answers in one unit

Every benchmark records both libraries' fitted parameters and evaluates **both** under
`hawk`'s unpenalized negative log-likelihood on the same events. That places two answers
from two different objectives into one common unit.

It is not a neutral unit. It scores `tick`'s answer with an objective `tick` was not
minimizing, and at `d > 1` was not even able to minimize. It is reported as what it is:
the reader can see how far the least-squares answer sits from the likelihood optimum,
which is the quantity that matters when deciding whether the timing comparison is
between two runs that reached comparable places.

**A timing comparison between runs that reached different optima is not a comparison.**
Where that happens it is said in place.

## 6. The grid, and why these sizes

Chosen from parameter counts and asymptotic cost, before any timing:

| Benchmark | `d` | parameters (`hawk`) | nominal `n` |
| --- | --- | --- | --- |
| `fit_d1` | 1 | 3 | 1e3, 1e4, 1e5, 1e6 |
| `fit_d10` | 10 | 111 | 1e4, 1e5, 1e6 |
| `fit_d100` | 100 | 10 101 | 1e5 |
| `simulate` | 1, 10 | — | 1e4, 1e5, 1e6 |

`fit_d1` repeats the positioning probe's grid so the two documents can be read against
each other.

`fit_d10` starts at `1e4`: at `1e3` there would be 9 events per parameter and the fit
would be reporting sampling noise.

`fit_d100` has a single point because it is squeezed from both sides. `hawk`'s gradient
pass is `O(n * d^2)`, which is `1e4` times the work per event that `d = 1` does, so
`n = 1e6` does not fit the §4.1 budget. Below `n = 1e5` there are fewer events than the
10 101 parameters being estimated, so the problem is not identified. One point is what
is left, and one point says nothing about scaling in `d` — which is stated rather than
papered over with a second point that would not mean anything.

## 7. Reproducing

```sh
benchmarks/suite/fit_d1.sh
benchmarks/suite/fit_d10.sh
benchmarks/suite/fit_d100.sh
benchmarks/suite/simulate.sh
```

Each is standalone, builds what it needs, creates its own virtual environment with the
pinned `tick`, and writes JSON to `benchmarks/results/`. `benchmarks/suite/create_diagrams.py`
regenerates every chart from the committed JSON with no manual step.

## 8. What these measurements will not establish

- Any parameter set other than the one in §3, or data not from `hawk`'s simulator.
- Time to a *given accuracy*. Both sides run to their own criteria, different in kind.
- Scaling in `d` beyond the three values measured, and nothing about `d = 100` scaling
  in `n` at all (§6).
- The cost of fitting `beta` separately from the rest. `tick` cannot fit it and `hawk`
  has no interface to hold it fixed.
- Statistical properties of either estimator: bias, variance, or efficiency.
- Anything about `tick` under settings other than those in §5. Its least-squares path is
  fast and this document does not claim otherwise.
