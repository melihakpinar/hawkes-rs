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

---

# Results

Run on the environment in §2. Raw output is committed at
`benchmarks/results/positioning-probe.json`.

## 7. Wall-clock time

Seconds. Median of 5 timed fits after 1 discarded warmup; `[min, max]` of the same 5.
The ratio column is `hawk median / tick median`, arithmetic only.

| nominal `n` | events | `hawk` median | `hawk` [min, max] | `tick` median | `tick` [min, max] | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1e3 | 889 | 0.001689 | [0.001258, 0.001699] | 0.004925 | [0.003556, 0.005466] | 0.34 |
| 1e4 | 10 079 | 0.009259 | [0.009232, 0.009476] | 0.003618 | [0.003537, 0.003649] | 2.56 |
| 1e5 | 100 041 | 0.099940 | [0.099595, 0.100087] | 0.027889 | [0.027821, 0.028042] | 3.58 |
| 1e6 | 998 244 | 1.003165 | [1.001432, 1.006357] | 0.272880 | [0.271477, 0.276625] | 3.68 |

`tick`'s median at `n = 1e3` (0.004925) is larger than its median at `n = 1e4`
(0.003618), and its spread at `n = 1e3` is the widest of any cell in the table. This is
recorded, not explained.

## 8. Parameters returned

`hawk` fits three parameters. `tick` fits two and is given `beta = 1.0`, the true value
(§5.1); its `beta` column is input, not output.

| nominal `n` | side | `mu` | `alpha` | `beta` |
| --- | --- | --- | --- | --- |
| 1e3 | `hawk` | 0.4946090535 | 0.5561509035 | 1.1158453050 |
| 1e3 | `tick` | 0.4775194357 | 0.5721140795 | 1.0 (fixed input) |
| 1e4 | `hawk` | 0.4874162896 | 0.6131595200 | 1.0002794165 |
| 1e4 | `tick` | 0.4874315257 | 0.6132065567 | 1.0 (fixed input) |
| 1e5 | `hawk` | 0.4985274500 | 0.6013546245 | 0.9899905845 |
| 1e5 | `tick` | 0.5003167681 | 0.5999290885 | 1.0 (fixed input) |
| 1e6 | `hawk` | 0.5003743247 | 0.5989976667 | 1.0083739829 |
| 1e6 | `tick` | 0.4988995462 | 0.6001803335 | 1.0 (fixed input) |

True parameters: `mu = 0.5`, `alpha = 0.6`, `beta = 1.0`.

## 9. Are they at the same optimum?

Both parameter vectors scored under `hawk`'s unpenalized negative log-likelihood on the
same data (§5.4). Lower is better under that objective. This scores `tick`'s answer
with an objective `tick` was not minimizing.

| nominal `n` | `hawk` nll | `tick` params, scored under `hawk`'s nll | difference (`tick` − `hawk`) |
| --- | --- | --- | --- |
| 1e3 | 654.292005 | 654.523646 | 0.231641 |
| 1e4 | 6066.352647 | 6066.352692 | 0.000044 |
| 1e5 | 62291.271514 | 62291.563149 | 0.291635 |
| 1e6 | 622971.850098 | 622973.906895 | 2.056797 |

The two sides do not return the same parameters and do not land on the same objective
value. Per §5.1 and §5.2 they are not minimizing the same function over the same
variables, so this is expected rather than a discrepancy to reconcile. **The timing
table in §7 therefore compares two different computations**, as set out in §5.

## 10. Convergence actually achieved

`hawk`, from its own instrumentation. `gradient norm` is the infinity norm of the
per-event log-space gradient at the returned point; the stopping threshold is `1e-6`
for the `converged` flag and the solver's own tolerance is `1e-10` (§5.3).

| nominal `n` | iterations | gradient norm | `converged` |
| --- | --- | --- | --- |
| 1e3 | 10 | 2.088e-9 | true |
| 1e4 | 10 | 5.477e-11 | true |
| 1e5 | 10 | 7.092e-12 | true |
| 1e6 | 10 | 6.440e-12 | true |

`tick`'s `HawkesExpKern` exposes no equivalent diagnostic through its public interface,
so no comparable column exists for it. Its iteration count and final gradient are not
recorded because they were not retrievable without reaching into internals.

## 11. What this measurement does not establish

Listed so the numbers are not read past their scope.

- Multivariate performance. `hawk` has no multivariate estimator; nothing here touches
  that case.
- Any `n` outside 1e3 to 1e6, any parameter set other than
  `(0.5, 0.6, 1.0)`, and any data not generated by `hawk`'s own simulator.
- Time to a *given accuracy*. Both sides ran to their own convergence criteria, which
  are different in kind (§5.3).
- Cost of fitting `beta`, separately from the rest. `tick` cannot fit it and `hawk` has
  no interface to hold it fixed, so the three-parameter and two-parameter problems
  could not be run on the same footing on either side.
- Memory, allocation behaviour, or startup cost.
- Accuracy, bias or variance of either estimator.

---

# Part 2: inner-loop cost, with the optimizer and the parameter count removed

Part 1 compared whole fits, and §5 recorded that the two sides solve different problems
with different optimizers and different stopping rules. This part measures one
objective evaluation, which removes all three of those.

Measurement only. No recommendation and no interpretation.

## 12. Methodology, part 2

Unchanged from §2-§4 except as noted: same machine, both native `arm64`,
single-threaded, median of 5 timed calls after 1 discarded warmup, identical event data
generated once by `hawk`'s simulator and read from disk by both sides.

### What is timed

| | |
| --- | --- |
| `hawk` | one `negative_log_likelihood_and_gradient` call |
| `tick` | one `ModelHawkesExpKernLogLik.loss_and_grad` call |

`loss_and_grad` is used rather than `loss()` then `grad()` because it is `tick`'s
single-pass entry point, matching `hawk`, which returns value and gradient from one
pass. Separate `loss()` and `grad()` timings are reported alongside.

Both are evaluated at the same point, the true parameters
`(mu, alpha, beta) = (0.5, 0.6, 1.0)`.

`n` is 1e4, 1e5, 1e6. The 1e3 case from part 1 is dropped: at that size the timed
region is a few microseconds and is dominated by call overhead.

### Two asymmetries remain

- **Gradient dimension.** `hawk` returns a 3-component gradient
  (`mu`, `alpha`, `beta`); `tick` returns 2 (`mu`, `alpha`), because it has no
  derivative with respect to the decay (§5.1).
- **`hawk`'s objective evaluation is not cheaper than its gradient evaluation.**
  `negative_log_likelihood` delegates to `negative_log_likelihood_and_gradient` and
  discards the gradient, so both cost one full pass. `tick`'s `loss()` and `grad()` are
  separate passes with different costs, both reported below.

## 13. Single evaluation, wall clock

Seconds. Median of 5 after 1 warmup; `[min, max]` of the same 5. `tick`'s figures are
**steady state**, i.e. with its weight cache already populated — see §15. The ratio
column is `hawk / tick loss_and_grad`, arithmetic only.

| nominal `n` | events | `hawk` value+grad | `hawk` [min, max] | `tick` `loss_and_grad` | `tick` [min, max] | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1e4 | 10 079 | 0.000403625 | [0.000379667, 0.000877875] | 0.000176042 | [0.000173500, 0.000177917] | 2.29 |
| 1e5 | 100 041 | 0.002907958 | [0.002676125, 0.003072792] | 0.001533417 | [0.001532834, 0.001546125] | 1.90 |
| 1e6 | 998 244 | 0.026231625 | [0.026192125, 0.026328583] | 0.015208333 | [0.015038167, 0.015213125] | 1.72 |

`tick`, `loss()` and `grad()` timed separately (steady state):

| nominal `n` | `loss()` | `grad()` | sum | `loss_and_grad` |
| --- | --- | --- | --- | --- |
| 1e4 | 0.000096750 | 0.000076417 | 0.000173167 | 0.000176042 |
| 1e5 | 0.000959000 | 0.000563167 | 0.001522167 | 0.001533417 |
| 1e6 | 0.009753583 | 0.005435959 | 0.015189542 | 0.015208333 |

## 14. Objective evaluations per fit

Evaluations, not iterations: line-search trials are counted.

`hawk`, from `Fit::objective_evaluations` and `Fit::gradient_evaluations`, added for
this measurement. `tick`, from `n_calls_loss`, `n_calls_grad` and `n_passes_over_data`
on the model object underlying `HawkesExpKern`, using the part 1 fit configuration.

| nominal `n` | `hawk` iters | `hawk` objective | `hawk` gradient | `hawk` total passes | `tick` `loss` | `tick` `grad` | `tick` passes over data |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e4 | 10 | 12 | 22 | 34 | 13 | 10 | 23 |
| 1e5 | 10 | 13 | 23 | 36 | 13 | 10 | 23 |
| 1e6 | 10 | 13 | 23 | 36 | 13 | 10 | 23 |

Per §12, each of `hawk`'s 34-36 passes computes value and gradient together. `tick`'s
23 passes are 13 value-only and 10 gradient-only, whose separate costs are in §13.

### Arithmetic cross-check against part 1

Passes times per-pass cost, against the part 1 fit medians. Multiplication only.

| nominal `n` | side | passes x per-pass | product | part 1 fit median |
| --- | --- | --- | --- | --- |
| 1e6 | `hawk` | 36 x 0.026231625 | 0.944 | 1.003165 |
| 1e6 | `tick` | 13 x 0.009753583 + 10 x 0.005435959 | 0.181 | 0.272880 |

## 15. The weight-cache hypothesis: confirmed, from source

The hypothesis was that `tick` precomputes and caches the `exp(-beta*d)` terms across
the optimization, because `beta` is fixed for it.

**This was established directly, not indirectly.** `tick`'s C++ is published; the
installed version has a matching tag, `v0.8.0.2`.

`lib/cpp/hawkes/model/model_hawkes_expkern_loglik_single.cpp` computes, for every
event, arrays `g` (excitation state) and `G` (compensator increments), holding the
exponential terms:

```cpp
const double ebt = std::exp(-decay * (t_i_k - t_i[k - 1]));
if (k < n_jumps_i)
  g_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * ebt;
G_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * (1 - ebt) / decay;
```

`lib/cpp/hawkes/model/base/model_hawkes_loglik_single.cpp` guards it with a flag, and
every entry point begins the same way:

```cpp
void ModelHawkesLogLikSingle::compute_weights() {
  ...
  weights_computed = true;
}
double ModelHawkesLogLikSingle::loss(const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  ...
}
```

`loss`, `grad`, `loss_i`, `grad_i`, `loss_and_grad` and `hessian_norm` all open with
`if (!weights_computed) compute_weights();`. So the exponentials are computed once,
lazily on first use, and reused by every later call.

`lib/include/tick/hawkes/model/model_hawkes_expkern_loglik_single.h` shows the cache is
invalidated when the decay changes:

```cpp
void set_decay(double decay) {
  this->decay = decay;
  weights_computed = false;
}
```

### Timing corroboration

The indirect test was run anyway, and agrees. Seconds, median of 5.

| nominal `n` | `set_data` | first `loss()` after `set_data` | subsequent `loss()` | ratio | first `loss()` after changing `decay` | steady `loss()` same object | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e4 | 0.000013833 | 0.000200459 | 0.000103188 | 1.94 | 0.000189959 | 0.000096875 | 1.96 |
| 1e5 | 0.000012875 | 0.001815042 | 0.000969270 | 1.87 | 0.001798459 | 0.000954833 | 1.88 |
| 1e6 | 0.000013250 | 0.024014625 | 0.009569563 | 2.51 | 0.024383750 | 0.009558125 | 2.55 |

`set_data` is ~13 microseconds at every `n`, including `n = 1e6`, so it does not do the
work. The first `loss()` after it is 1.9-2.5x the subsequent ones, and mutating `decay`
restores that penalty on the next call — matching `set_decay` clearing
`weights_computed`.

### Consequence for §13

`tick`'s §13 figures are steady-state, so they exclude the cached work. The cold figure
is the "first `loss()`" column above. At `n = 1e6` that is 0.024014625 against `hawk`'s
0.026231625.

## 16. What part 2 does not establish

- Memory. `tick`'s cache is `g` and `G` over all events; its size is not measured here,
  and neither is `hawk`'s allocation behaviour.
- Whether either evaluation could be faster. Only what they currently cost was measured.
- Anything about multivariate models, other parameter values, other `n`, or data not
  generated by `hawk`'s simulator.
- Accuracy. Both were evaluated at the same point; the values returned were not
  compared here. Part 1 §9 covers agreement of fitted parameters.
- Why the ratio in §13 moves with `n`.
