# Univariate exponential-kernel Hawkes: log-likelihood

Status: **awaiting owner approval.** Per CLAUDE.md §4 no implementation may proceed
until this is approved.

Scope: univariate, exponential kernel, one observation window `[0, T]`.

## 0. Citations, and one honest gap

Every convention used here is pinned in `conventions.md` by experiment, not by
prose. The symbols are cited as follows:

| Symbol / choice | Source |
| --- | --- |
| kernel `phi(s) = alpha*beta*exp(-beta s)` | `conventions.md` C1, experiments E1, E2 |
| branching ratio `= alpha` | `conventions.md` C2, `HawkesKernelExp.get_norm()` |
| strict bounds, predictable intensity | `conventions.md` C3, experiment E2b |
| compensator integrates to `T` | `conventions.md` C4 |
| `T` supplied by the caller | `conventions.md` C5 |
| ties, ordering, input contract | `conventions.md` C8, experiment E3 |

**The gap.** CLAUDE.md §2 requires papers to be cited by equation number, and
`docs/references/` is empty — the PDFs are not committed and I do not have them. I
will not invent equation numbers I have not read. Two consequences:

- The recursion in §4 is attributed to **[Ozaki1979]** for provenance, *without* an
  equation number. The derivation below is self-contained: every step is shown, so
  it can be checked line by line without the paper.
- The likelihood in §3 is the standard point-process likelihood. It is derived here
  from the compensator rather than quoted.

Filed as **OQ-11**. Before this document is treated as fully compliant with
CLAUDE.md §2, the PDFs must be added and the equation numbers filled in. The
mathematics does not depend on that; the audit trail does.

## 1. Setup and notation

Parameters, all strictly positive:

| Symbol | Meaning |
| --- | --- |
| `mu` | baseline intensity |
| `alpha` | branching ratio; the integral of the kernel (C1, C2) |
| `beta` | exponential decay rate |
| `T` | observation horizon, supplied by the caller (C5) |

Data: `n` event times `t_1 <= t_2 <= ... <= t_n`, all in `[0, T]` (C8). Ties are
permitted, so the inequalities are **not** strict.

Stationarity requires `alpha < 1` (C2). It is **not** enforced during optimization;
it is reported as a diagnostic on the fit result (CLAUDE.md §6).

## 2. Intensity and compensator

**Kernel** (C1), zero for non-positive lag:

```
phi(s) = alpha * beta * exp(-beta * s)   for s > 0
phi(s) = 0                               for s <= 0
```

**Conditional intensity** (C3). The sum is over event times **strictly** less than
`t`, which makes `lambda` predictable (left-continuous):

```
lambda(t) = mu + sum_{i : t_i < t} alpha * beta * exp(-beta * (t - t_i))
```

Two consequences worth stating because they are exactly what index errors get wrong:

- `lambda(t_1) = mu`. The first event sees no excitation.
- If `t_i = t_k` for `i != k`, neither contributes to the other's intensity. The
  condition is on *times*, not on array positions (C8, experiment E3b).

**Compensator.** Integrate the intensity over the window. Every `t_i` lies in
`[0, T]`, so each kernel contributes over `[t_i, T]`:

```
Lambda(T) = int_0^T lambda(u) du
          = mu*T + sum_{i=1}^{n} int_0^T phi(u - t_i) du
          = mu*T + sum_{i=1}^{n} int_{t_i}^{T} alpha*beta*exp(-beta*(u - t_i)) du
```

Substituting `v = u - t_i` in the inner integral:

```
int_{t_i}^{T} alpha*beta*exp(-beta*(u-t_i)) du
    = int_0^{T-t_i} alpha*beta*exp(-beta*v) dv
    = alpha * [ -exp(-beta*v) ]_0^{T-t_i}
    = alpha * ( 1 - exp(-beta*(T - t_i)) )
```

giving

```
Lambda(T) = mu*T + alpha * sum_{i=1}^{n} ( 1 - exp(-beta*(T - t_i)) )      (2.1)
```

This is checkable against `tick` without any likelihood:
`HawkesKernelExp(alpha, beta).get_primitive_value(s)` returns
`alpha*(1 - exp(-beta*s))`, confirmed at `s` in {0.5, 1.0, 3.0} in experiment E1.

Note the integral runs to `T`, not to `t_n` (C4). When `T > t_n` the difference is
not a constant — it grows with the trailing dead time.

## 3. Log-likelihood

For a point process on `[0, T]` with predictable intensity `lambda`, observed events
`t_1..t_n`:

```
log L = sum_{k=1}^{n} log lambda(t_k) - Lambda(T)                          (3.1)
```

`hawk` minimizes the **negative** log-likelihood:

```
nll = Lambda(T) - sum_{k=1}^{n} log lambda(t_k)                            (3.2)
```

Combining (2.1) and (3.2):

```
nll(mu, alpha, beta)
    = mu*T
    + alpha * sum_{i=1}^{n} ( 1 - exp(-beta*(T - t_i)) )
    - sum_{k=1}^{n} log( mu + sum_{i : t_i < t_k} alpha*beta*exp(-beta*(t_k - t_i)) )
                                                                           (3.3)
```

(3.3) is the **definition**. It is `O(n^2)` because of the inner sum. M1 Part B
step 5 transcribes exactly this, with no simplification, as the reference oracle.

`lambda(t_k) >= mu > 0` always, so the logarithm is never evaluated at zero or below
and no guard is needed.

## 4. The O(n) recursion

Define the **excitation state**

```
A(t) := sum_{i : t_i < t} exp(-beta * (t - t_i))                           (4.1)
```

so that `lambda(t) = mu + alpha*beta*A(t)`. The recursion is due to [Ozaki1979]
(equation number pending, §0).

### 4.1 Why the textbook recursion is wrong here

The usual statement is `A_1 = 0` and

```
A_k = exp(-beta*(t_k - t_{k-1})) * ( 1 + A_{k-1} )                         (4.2)
```

**(4.2) is only valid when all timestamps are distinct.** If `t_k = t_{k-1}` then
`exp(0) = 1` and (4.2) yields `A_k = 1 + A_{k-1}`, which counts `t_{k-1}` as
exciting `t_k`. C3 and experiment E3b say it must not. Since `hawk`'s input contract
admits ties (C8), (4.2) cannot be the expression that gets coded.

### 4.2 The recursion that is correct with ties

Group by **distinct** time. Let

```
s_1 < s_2 < ... < s_m           the distinct event times
c_j = #{ i : t_i = s_j }        the multiplicity of s_j,   sum_j c_j = n
```

and define `B_j := A(s_j)`. Because `lambda` depends on `t` only through `A(t)`, all
`c_j` events at `s_j` share one intensity `lambda_j := mu + alpha*beta*B_j`.

**Base case.** No event lies strictly before the earliest event time, so

```
B_1 = 0                                                                    (4.3)
```

**Step.** For `j >= 2`, write `d_j := s_j - s_{j-1} > 0`. The events strictly before
`s_j` are exactly the events strictly before `s_{j-1}`, plus the `c_{j-1}` events at
`s_{j-1}`:

```
B_j = sum_{i : t_i < s_j} exp(-beta*(s_j - t_i))

    = sum_{i : t_i < s_{j-1}} exp(-beta*(s_j - t_i))
    + sum_{i : t_i = s_{j-1}} exp(-beta*(s_j - s_{j-1}))

    = exp(-beta*d_j) * sum_{i : t_i < s_{j-1}} exp(-beta*(s_{j-1} - t_i))
    + c_{j-1} * exp(-beta*d_j)

    = exp(-beta*d_j) * ( B_{j-1} + c_{j-1} )                               (4.4)
```

With all `c_j = 1` and `m = n`, (4.4) collapses to (4.2), so this generalizes the
textbook recursion rather than replacing it.

### 4.3 The likelihood in grouped form

Both sums in (3.3) group the same way:

```
nll(mu, alpha, beta)
    = mu*T
    + alpha * sum_{j=1}^{m} c_j * ( 1 - exp(-beta*(T - s_j)) )
    - sum_{j=1}^{m} c_j * log( mu + alpha*beta*B_j )                       (4.5)
```

with `B_j` from (4.3)-(4.4). One pass, `O(n)`.

## 5. The exact expression to be coded

Inputs: sorted `t[0..n-1]` in `[0, T]`, and `mu, alpha, beta > 0`.

```
if n == 0:
    return mu * T                      # no events: compensator only

compensator_excitation = 0.0           # accumulates sum_j c_j * (1 - exp(-beta*(T-s_j)))
log_term                = 0.0          # accumulates sum_j c_j * log(lambda_j)
B                       = 0.0          # B_j, the excitation state at the current s_j
previous_time           = t[0]         # s_{j-1}
previous_count          = 0            # c_{j-1}; 0 marks "no previous distinct time"

for k in 0 .. n-1:
    if t[k] != previous_time:                       # a new distinct time s_j
        d = t[k] - previous_time                    # d_j > 0
        B = exp(-beta * d) * (B + previous_count)   # (4.4)
        previous_time  = t[k]
        previous_count = 0
    # every event at this distinct time shares one intensity
    intensity = mu + alpha * beta * B               # lambda_j
    log_term += ln(intensity)
    compensator_excitation += -expm1(-beta * (T - t[k]))
    previous_count += 1

return mu * T + alpha * compensator_excitation - log_term
```

The loop visits each event once and accumulates `c_j` implicitly through
`previous_count`, so the grouping of §4.2 costs nothing.

### Numerical notes (CLAUDE.md §6)

- `1 - exp(-x)` is written `-expm1(-x)`. For small `beta*(T - t_i)` — every event
  near the end of the window — the direct form loses precision to cancellation,
  because `exp(-x)` rounds to a value very close to 1. This matters: in a long
  window, many events sit close to `T`.
- `ln(intensity)` needs no `ln_1p`. `intensity >= mu`, which is bounded away from
  zero by the positivity constraint, so the argument is never near 1 in a way that
  costs precision, and never near 0.
- `exp(-beta * d)` with `d > 0` and `beta > 0` is in `(0, 1)`; it underflows to `0`
  for large `beta*d`, which is the correct limit — an event long past has no
  influence. No guard needed.
- `f64` throughout.

### Comparison with the definition

Part B step 7 must show (4.5) agrees with (3.3) to `1e-12` over randomized
parameters and event counts. Both evaluate the same quantity in the same arithmetic
and differ only in summation order, so a larger gap is a real defect, not
accumulation.

## 6. Edge cases

| Case | Behaviour |
| --- | --- |
| `n == 0` | `nll = mu*T`. No log term. |
| `n == 1` | `B_1 = 0`, so `lambda = mu`; `nll = mu*T + alpha*(1 - exp(-beta*(T-t_1))) - ln(mu)`. |
| all timestamps tied | `m = 1`, `B_1 = 0`; every event has `lambda = mu`. |
| `t_n == T` | Allowed (C8). Its compensator contribution is `alpha*(1 - exp(0)) = 0`. |
| `t_1 == 0` | Allowed (C8). |
| unsorted input | Rejected as an error (C8), not silently sorted. |
| `T < t_n` | Rejected as an error. |

## 7. What this is checked against

1. **Brute force** (Part B step 5) — (3.3) transcribed directly, `O(n^2)`. Primary
   gate for (4.5), to `1e-12`. Uses no `tick`.
2. **Finite-difference gradient** (Part B step 8) — the M0 harness, against
   `univariate_gradient.md`.
3. **Time-rescaling residuals** (Part B step 6) — validates the compensator and the
   simulator jointly.
4. **`tick`** — last, and only after the above. The identity to test is
   `nll == tick_loss * n_jumps + D*T` (OQ-8). Preliminary evidence gathered in Part A
   confirms it holds at machine precision on all six committed fixtures across all 24
   parameter points, 18 of which have `alpha != 0`; Part B step 9 confirms it with
   `hawk`'s own implementation.

## 8. Numerical check run during Part A

The expressions above were checked before being handed over, so that what is being
approved is known-consistent rather than merely plausible. The script is
`docs/derivations/check_univariate_derivation.py` — a throwaway reference
implementation in Python, deliberately **not** `hawk` code, run with `python3` and no
dependencies.

454 cases: 300 with randomized distinct timestamps (`n` from 0 to 40), 150
tie-heavy, plus the degenerate cases from §6 (empty, single, all-tied, endpoints).

```
worst relative |recursive (4.5) - brute force (3.3)| : 3.553e-15
worst relative |analytic gradient - central diff|    : 1.828e-08
```

The first is at the `f64` round-off floor, three orders inside the `1e-12` gate that
Part B step 7 must meet.

**The tie case is not hypothetical.** On `t = [1.0, 2.0, 2.0, 3.0]` with
`mu=0.7, alpha=0.5, beta=1.3, T=5`:

| expression | value |
| --- | --- |
| definition (3.3) | 5.961059318008664 |
| grouped recursion (4.5) | 5.961059318008664 |
| textbook recursion (4.2) | 5.406576697862245 |

The textbook form is wrong by about 9%, silently, on input `hawk`'s contract accepts.
This is the concrete justification for §4.2 and for C8's decision to admit ties.

This check does **not** replace Part B. It is Python, it is not sabotage-tested, and
it shares an author with the derivation. Part B step 5 and step 7 redo it in Rust
with the M0 harnesses.
