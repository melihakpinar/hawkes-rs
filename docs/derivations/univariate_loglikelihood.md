# Univariate exponential-kernel Hawkes: log-likelihood

Status: **awaiting owner approval.** Per CLAUDE.md §4 no implementation may proceed
until this is approved.

Scope: univariate, exponential kernel, one observation window `[0, T]`.

## 0. Citations

Primary citable reference, per CLAUDE.md §2: **[Laub2015]** — Laub, Taimre & Pollett,
*Hawkes Processes*, arXiv:1507.02822v1, in `docs/references/`. Freely accessible, so
every equation below resolves to something a reader can check.

| Symbol / step | Source |
| --- | --- |
| exponential-kernel intensity, strict bounds | [Laub2015, eq. 4] |
| branching ratio | [Laub2015, eq. 5] |
| stationary mean intensity | [Laub2015, eq. 6] |
| likelihood on `[0, T]` | [Laub2015, Theorem 3] (from Daley & Vere-Jones, Prop. 7.2.III) |
| log-likelihood | [Laub2015, eq. 17] |
| compensator, exponential kernel | [Laub2015, eq. 18] |
| `O(n^2)` direct form | [Laub2015, eq. 19] |
| `O(n)` recursion | [Laub2015, eq. 20], attributed there to Ozaki [Ozaki1979] |
| `O(n)` log-likelihood | [Laub2015, eq. 21] |
| kernel normalization actually used | `conventions.md` C1, experiments E1, E2 |
| predictable intensity | `conventions.md` C3, experiment E2b |
| compensator integrates to `T` | `conventions.md` C4 |
| `T` supplied by the caller | `conventions.md` C5 |
| ties, ordering, input contract | `conventions.md` C8, experiment E3 |

[Ozaki1979] is cited for provenance only, without an equation number: no PDF of it is
in `docs/references/`, and CLAUDE.md §2 permits provenance-only citation in that case.
Everything it is credited with here is reproduced in [Laub2015] with equation numbers.

### 1.1 Reparametrization: [Laub2015] is not in `tick`'s parametrization

This is the single most important line in this document, and getting it wrong makes
every equation below wrong by a factor of `beta`.

[Laub2015, eq. 4] writes

```
lambda*(t) = lambda + sum_{t_i < t} alpha_L * exp(-beta*(t - t_i))
```

`hawk` uses `tick`'s parametrization (`conventions.md` C1), in which the kernel is
`alpha*beta*exp(-beta t)`. So throughout this document:

```
lambda  ->  mu           (Laub's background intensity)
alpha_L ->  alpha*beta   (Laub's jump size)
beta    ->  beta         (unchanged)
```

Under that substitution [Laub2015, eq. 5]'s branching ratio `alpha_L/beta` becomes
`alpha`, and [Laub2015, eq. 6]'s stationary mean `lambda/(1-n)` becomes
`mu/(1-alpha)`. Both agree with `tick` (C2), which is a useful cross-check that the
substitution is the right way round.

### 1.2 One more difference from [Laub2015]

Eq. 18-21 are derived in [Laub2015] §4.2 for a process *observed up to the last
arrival*, so their horizon is `t_k`, not `T`. Theorem 3 is the general statement with
`[0, T]`. `hawk` takes `T` from the caller and never infers it (C5), so §2-§4 below
carry `T` throughout. Setting `T = t_n` recovers Laub's form.

Transcribing eq. 21 literally with a caller-supplied `T > t_n` would silently drop
`int_{t_n}^{T} lambda*(u) du`. That is CLAUDE.md §1.3's "compensator on the tail"
hazard, and it appears here as a real difference between the cited paper and what
must be coded — not as a hypothetical.

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

[Laub2015, Theorem 3], which the paper attributes to Daley & Vere-Jones
Proposition 7.2.III, gives the likelihood of a regular point process on `[0, T]`:

```
L = [ prod_{i=1}^{n} lambda*(t_i) ] * exp( -int_0^T lambda*(u) du )
```

Taking logarithms, and using `Lambda(T) = int_0^T lambda*(u) du` — this is
[Laub2015, eq. 17] with the horizon generalized from `t_k` to `T` per §1.2:

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

### 3.1 With ties, (3.3) is not a likelihood

[Laub2015, Theorem 3] is the likelihood of a **simple** point process. Its §4.1 proof
uses that assumption directly — *"The HP is a simple point process, meaning that
multiple arrivals cannot occur at the same time"* — to set `F*(t_k) = 0` and obtain
eq. 14.

In a simple point process the probability of two simultaneous arrivals is zero. So on
data containing exact ties, which `hawk`'s input contract admits (C8), (3.3) is **not
a likelihood**. It is a formal extension of one: the expression still evaluates, the
intensity path is still well defined, and it still agrees with `tick` (see §7), but
it is no longer the density of anything the model can generate.

This is a statement about interpretation, not a defect, and it has consequences that
must not be discovered later:

- **The maximum-likelihood estimator's guarantees do not carry over.** Consistency,
  asymptotic normality and efficiency — proved for this estimator by Ogata, per
  [Laub2015] §4.3 — are results about the likelihood of a simple point process. On
  tied data the estimator is still a well-defined M-estimator, and it is not a
  maximum-likelihood estimator in the sense those theorems require. `hawk` must not
  advertise MLE asymptotics for tied input.
- **The round-trip property test (Part B step 10) must not generate ties.** Parameters
  are not recoverable from data the model assigns probability zero to, so a tie-
  generating strategy would produce a failure that looks like an estimator defect and
  is not one. Ogata thinning (Part B step 6) generates continuous inter-arrival times
  and will not produce ties on its own; the hazard is a `proptest` strategy that
  synthesises timestamps directly. **The test's doc comment must say why ties are
  excluded**, or the exclusion will look arbitrary and be removed by someone tidying up.
- Ties remain supported for *evaluation* — computing the objective, the gradient, and
  the compensator on real data whose clock resolution produced them. What is withheld
  is the statistical interpretation, not the arithmetic.

(3.3) is the **definition**, and is [Laub2015, eq. 19] under the substitution of
§1.1 and with the horizon of §1.2. It is `O(n^2)` because of the inner sum. M1 Part B
step 5 transcribes exactly this, with no simplification, as the reference oracle.

Sanity check against the cited form. [Laub2015, eq. 18] gives the compensator as
`Lambda(t_k) = lambda*t_k - (alpha_L/beta) * sum_i [ exp(-beta*(t_k - t_i)) - 1 ]`.
Substituting `alpha_L = alpha*beta` and `t_k -> T` turns `alpha_L/beta` into `alpha`
and flips the bracket's sign, giving
`mu*T + alpha * sum_i ( 1 - exp(-beta*(T - t_i)) )`, which is (2.1). The `beta` in the
numerator of the substitution cancels the `beta` in Laub's denominator — the clearest
available confirmation that §1.1 is the right way round.

`lambda(t_k) >= mu > 0` always, so the logarithm is never evaluated at zero or below
and no guard is needed.

## 4. The O(n) recursion

Define the **excitation state**

```
A(t) := sum_{i : t_i < t} exp(-beta * (t - t_i))                           (4.1)
```

so that `lambda(t) = mu + alpha*beta*A(t)`. Under §1.1's substitution this is exactly
[Laub2015, eq. 20]'s `A(i)`, and (4.5) below is [Laub2015, eq. 21]. The recursion is
attributed there to [Ozaki1979].

### 4.1 Why the textbook recursion is wrong here

[Laub2015, eq. 20] states, with base case `A(1) = 0`:

```
A_k = exp(-beta*(t_k - t_{k-1})) * ( 1 + A_{k-1} )                         (4.2)
```

**(4.2) is only valid when all timestamps are distinct — and [Laub2015] says so.**
Its §4.1 proof states plainly: *"The HP is a simple point process, meaning that
multiple arrivals cannot occur at the same time."* Eq. 20 is derived inside that
assumption, so using it on tied data is outside the domain the paper claims for it.
This is not a defect in [Laub2015]; it is a defect in transcribing eq. 20 without
carrying its hypothesis across.

If `t_k = t_{k-1}` then
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
- `ln(intensity)` is computed with plain `ln`, **not** `ln_1p`. The reason is not
  that `intensity` stays away from 1 — it does not, `mu = 1.0` is an ordinary
  parameter value and then `intensity` sits just above 1 whenever the excitation
  state is small. `ln_1p(x)` is only worth reaching for when `x = intensity - 1` is
  itself available to full precision. Here `intensity` is formed as
  `mu + alpha*beta*B`, so computing `intensity - 1` would round exactly as badly as
  `intensity` did: the subtraction cannot recover information the addition already
  discarded. `ln_1p` would buy nothing and would obscure the expression. What does
  matter is that `intensity >= mu > 0`, so the argument is never zero or negative.
- `exp(-beta * d)` with `d > 0` and `beta > 0` is in `(0, 1)`; it underflows to `0`
  for large `beta*d`, which is the correct limit — an event long past has no
  influence. No guard needed.
- `f64` throughout.

### Comparison with the definition

Part B step 7 must show (4.5) agrees with (3.3) to `1e-12` over randomized
parameters and event counts. Both evaluate the same quantity in the same arithmetic
and differ only in summation order, so a larger gap is a real defect — **but that
claim needs a bound on `n`, and it needs the tolerance to be relative.**

`docs/derivations/check_summation_scaling.py` measures it, using `math.fsum` (exactly
rounded) as ground truth so each side's own error is visible rather than only their
difference. Relative to `|nll|`:

| `beta` | `n` | `|nll|` | recursion vs exact | brute force vs exact | recursion vs brute |
| --- | --- | --- | --- | --- | --- |
| 0.90 | 2000 | 727 | 0.0 | 0.0 | 0.0 |
| 0.90 | 8000 | 2876 | 6.3e-16 | 0.0 | 6.3e-16 |
| 0.90 | 20000 | 7205 | 1.0e-15 | 0.0 | 1.0e-15 |
| 0.05 | 2000 | 666 | 6.8e-15 | 0.0 | 6.8e-15 |
| 0.05 | 8000 | 2680 | 1.0e-15 | 0.0 | 1.0e-15 |
| 0.05 | 20000 | 6699 | 1.3e-14 | 0.0 | 1.3e-14 |

Two things this shows, and neither is what a first guess would predict.

**The `O(n^2)` brute force is not the error-accumulating side.** Its error against
`fsum` is exactly zero at every `n` tested, including `beta = 0.05`, where the kernel
decays slowly. The reason is structural: the inner sum's terms are
`exp(-beta*(t_k - t_j))`, which fall off geometrically, so however long the sum
formally is, only `O(1/(beta*inter-arrival))` terms are numerically significant. The
sum's *effective* length does not grow with `n`. A naive `n^2` summation would
accumulate error, but this particular `n^2` summation does not have `n^2` significant
terms. **Kahan summation is therefore not needed**, and adding it would be
unfalsifiable ceremony.

**The recursion is the limiting side**, at `1.3e-14` relative in the worst case
measured. It carries `B_j` forward across the whole sequence, so its error does
propagate — damped by `exp(-beta*d_j)` at every step, which is why growth is slow and
noisy rather than monotone.

Consequences for the gate:

1. **The tolerance is relative, not absolute.** `|nll|` grows roughly linearly in `n`
   — 7205 at `n = 20000` — so an absolute `1e-12` gate would demand `1.8e-19`
   relative and fail on correct code. At `n = 20000` the observed absolute
   discrepancy is about `9e-11`, already a hundredfold over an absolute `1e-12`.

   **But the denominator must not be `|nll|`.** From (4.5),

   ```
   nll = mu*T + alpha*compensator_excitation - log_term
   ```

   is a difference of large terms, and it passes through zero. At parameter points
   where `mu*T + alpha*compensator_excitation ~= log_term` the result is near zero
   while the quantities actually being summed stay large, so a relative error taken
   against `|nll|` diverges on correct code. A randomized parameter sweep enters that
   region eventually — it is not a corner case, it is a surface.

   The gate therefore measures error relative to the **scale of the computation**,
   not to the size of its result:

   ```
   scale = mu*T
         + alpha * compensator_excitation           # both non-negative
         + sum_j c_j * |log(lambda_j)|              # magnitudes, not the signed sum

   |recursive - brute| <= 1e-12 * scale
   ```

   This is what floating-point error is actually proportional to: the magnitudes fed
   into the accumulators, independent of how much they cancel at the end.

   Note the third term is `sum |log lambda_j|`, not `|sum log lambda_j|`. The signed
   sum has the same defect one level down — `log lambda_j` is negative wherever
   `lambda_j < 1`, which is routine for `mu < 1`, so a `log_term` near zero can be the
   cancellation of thousands of `O(1)` contributions. Summing magnitudes measures the
   accumulator traffic honestly.

   All three terms are available for free: the implementation already accumulates the
   first two, and the third costs one `abs` per event in the test's reference path.
   **The test's doc comment must carry this reasoning**, or the unusual denominator
   reads as an accident and gets "simplified" back to `|nll|`.
2. **The comparison test is bounded at `n <= 50000`.** At the measured worst case of
   `1.3e-14` relative for `n = 20000`, and with the error growing no faster than
   `sqrt(n)`, `n = 50000` predicts about `2e-14` — roughly 50x inside the gate.
   Reaching `1e-12` would need `n` of order `10^8`, which no test will run. The bound
   exists so the claim is checked rather than assumed, not because failure is near.
3. **This argument is specific to a geometrically decaying kernel.** If a heavy-tailed
   kernel is ever added — power-law is out of scope for v0.1.0 (CLAUDE.md §8) — the
   effective-length argument fails, the brute force becomes the limiting side, and
   both the bound and the no-Kahan conclusion must be re-derived. Recorded here so
   that whoever adds one finds the reasoning rather than the conclusion.

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

   **The identity also survives exact ties**, which was not obvious and was checked
   rather than assumed (`benchmarks/docker/tie_identity.py`). The concern was that
   `tick` might resolve ties by array index — the textbook recursion (4.2) — in which
   case the identity would have to break on tied data and the break would not be a
   defect. It does not. Comparing `tick` against both semantics:

   | case | delta, time-based (ours) | delta, index-based |
   | --- | --- | --- |
   | no ties | 0.0 | 0.0 |
   | one tied pair | 0.0 | 3.9e-1 |
   | triple tie | -1.8e-15 | 1.1e0 |
   | ties at both ends | -8.9e-16 | 8.3e-1 |
   | bivariate, cross-component tie | -1.8e-15 | n/a |

   `tick` resolves ties by **time**, matching §4.2's grouped recursion. Note this does
   not contradict experiment E3a, where three orderings of the same three timestamps
   gave three different losses: that experiment used *distinct* timestamps and shows
   `tick` requires sorted input. Among *equal* timestamps there is no order to get
   wrong.

   So a tied fixture may be added to the corpus without the differential test being
   expected to fail. What a tied fixture would **not** validate is the statistical
   interpretation — see §3.1.

## 8. Numerical check run during Part A

The expressions above were checked before being handed over, so that what is being
approved is known-consistent rather than merely plausible. The script is
`docs/derivations/check_univariate_derivation.py` — a throwaway reference
implementation in Python, deliberately **not** `hawk` code, run with `python3` and no
dependencies.

454 cases: 300 with randomized distinct timestamps (`n` from 0 to 40), 150
tie-heavy, plus the degenerate cases from §6 (empty, single, all-tied, endpoints).

```
worst relative |recursive (4.5) - brute force (3.3)|            : 3.553e-15
worst relative |analytic gradient - central diff|               : 1.828e-08
worst relative |-nll - Laub eq.21 with alpha_L = alpha*beta|    : 1.170e-15
```

The third line checks §1.1's reparametrization against the cited paper directly: over
200 distinct-timestamp cases, `-nll` evaluated at `T = t_n` reproduces
[Laub2015, eq. 21] to machine precision. It also checks that the substitution is
*discriminating* rather than vacuous — using `alpha_L = alpha/beta` instead gives
`-4.900096161055312` where the correct direction gives `-5.743925007617449` on the
same input. A factor-of-`beta` slip would not pass unnoticed.

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
