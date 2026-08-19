# Positioning probe: `hawk` vs `tick`, univariate fit wall time

Measurement only. No recommendation and no interpretation is offered here.

**The methodology below was fixed and committed before any timing was run.** The
results section was appended afterwards and nothing in the methodology was changed
once numbers existed.

---

## 1. What is being compared

Wall-clock time of one univariate maximum-likelihood fit, on identical event data, for
`n` in {1e3, 1e4, 1e5, 1e6}.

Nothing else is measured: not memory, not simulation, not data loading, not accuracy.

## 2. Environment

| | |
| --- | --- |
| Machine | Apple M2, 8 cores, 16 GiB |
| Architecture | `arm64` (native on both sides) |
| OS | macOS, Darwin 22.6.0 |
| `hawk` | this working tree, `cargo build --release` |
| `rustc` | 1.95.0 |
| Python | 3.14.5 |
| `tick` | 0.8.0.2 |
| `numpy` / `scipy` | 2.5.2 / 1.18.0 |
| Threads | 1 on both sides (see below) |

### Both sides run native on the same machine

`tick` 0.8.0.2 publishes a `macosx_11_0_arm64` wheel for CPython 3.14, so it is
installed natively rather than in the pinned `linux/amd64` Docker image used for
fixtures. **This is a deliberate departure from `benchmarks/docker/`.** That image
exists so committed fixtures are byte-reproducible, and it runs under emulation on
this host. Emulated timings compared against native ones would measure the emulator.

The pinned image remains the oracle for correctness; it is not used for timing.

### Threads

`hawk` is single-threaded throughout — it has no parallelism at all.

The `tick` process is run with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1`.
`HawkesExpKern` exposes no `n_threads` argument, so the environment is the only
control available.

## 3. Data

Generated once per `n` by `hawk`'s simulator (Ogata thinning), seeded, written to disk
as one `f64` per line in Rust's shortest-round-trip decimal form, and read back by both
sides. Both therefore fit **exactly the same events**.

Simulation time is not measured, and the file is read before the clock starts on each
side.

| | |
| --- | --- |
| True parameters | `mu = 0.5`, `alpha = 0.6`, `beta = 1.0` |
| Stationary mean intensity | `mu / (1 - alpha) = 1.25` |
| Horizon | `T = n / 1.25`, so the expected count is `n` |
| Seed | 20260819, fixed |

The realized event count differs from the nominal `n` because simulation is random; the
realized count is reported.

## 4. Timing protocol

- 1 warmup fit, discarded.
- 5 timed fits.
- **Median** of the 5 reported. Minimum and maximum also reported, so dispersion is
  visible.
- `std::time::Instant` on the `hawk` side, `time.perf_counter()` on the `tick` side.
  Both are monotonic wall clocks.
- The clock covers the fit call only. Data loading, model construction that can be
  hoisted, and result extraction are outside it on both sides.

## 5. What could not be equalized

Three differences are structural. They are recorded here rather than adjusted away,
because adjusting them would mean measuring something other than what each library
does.

### 5.1 `tick` does not fit `beta`; `hawk` does

`ModelHawkesExpKernLogLik(decay, n_threads)` and `HawkesExpKern(decays, ...)` both take
the decay as a **fixed constructor argument**. `tick` has no univariate exponential-kernel
estimator that treats it as a free parameter — `HawkesExpKern.decays` echoes back what
was passed in.

So:

- `hawk` solves a **3-parameter** problem: `(mu, alpha, beta)`.
- `tick` solves a **2-parameter** problem: `(mu, alpha)`, with `beta` supplied.

`tick` is given the **true** `beta = 1.0`, which is the most favourable case for it:
it is handed for free the parameter the other side has to find.

These are different optimization problems. Any time difference includes this.

### 5.2 `tick`'s objective carries an L2 penalty; `hawk`'s does not

`HawkesExpKern(penalty="none")` raises `ValueError: BFGS only accepts ProxZero and
ProxL2sq for now`, and the `gd`/`agd` solvers fail with `RuntimeError: The sum of the
influence on someone cannot be negative`, because `ProxZero` admits negative
coefficients that the C++ model rejects. Unpenalized likelihood is therefore not
reachable through this interface.

`C = 1e9` is used instead. The penalty strength is `1/C`, and the fitted parameters are
numerically indistinguishable from a weaker penalty still:

| `C` | fitted `mu` | fitted `alpha` |
| --- | --- | --- |
| 1e3 | 0.5455938971 | 0.5489748957 |
| 1e6 | 0.5460860266 | 0.5491667884 |
| 1e9 | 0.5460865199 | 0.5491669800 |
| 1e12 | 0.5460865204 | 0.5491669802 |

`C = 1e9` and `C = 1e12` agree to nine significant figures, so at `C = 1e9` the penalty
is not what determines the answer. It is still not literally the same objective.

### 5.3 The stopping criteria are not the same kind of criterion

They cannot be equalized, only reported:

| | `hawk` | `tick` |
| --- | --- | --- |
| Solver | L-BFGS (`argmin` 0.11), More-Thuente line search, 7 correction pairs | `HawkesExpKern` with `solver="bfgs"` |
| Objective | negative log-likelihood **per event**, in log-parameter space | L2-penalized likelihood, natural parameters, `ProxL2sq` |
| Stop | gradient tolerance `1e-10` on the per-event log-space gradient | `tol = 1e-10` |
| Iteration cap | 500 | 500 |

`hawk` stops on a gradient norm; `tick`'s `tol` is passed into its own solver's
criterion, which is not the same quantity. **Times measured under different
convergence criteria are not directly comparable.** Both are set as tight as their
interfaces allow, and the achieved objective is reported below so it is visible whether
each actually reached an optimum.

### 5.4 Comparability of the reported optima

Because of 5.1 and 5.2 the two sides are minimizing different functions, and are not
expected to return the same parameters.

To make the difference measurable rather than assumed, the results record, for both
parameter vectors, the **unpenalized negative log-likelihood evaluated by `hawk` on the
same data**. That places both answers in one common unit. Note this scores `tick`'s
answer under `hawk`'s objective, which is not the objective `tick` was minimizing.

## 6. Reproducing

```sh
benchmarks/suite/positioning_probe.sh
```

It builds `hawk` in release, generates the data, runs both sides, and writes the raw
numbers to `benchmarks/results/positioning-probe.json`.

`tick` must be installed natively; the script prints the required `pip install` if it
is missing.
