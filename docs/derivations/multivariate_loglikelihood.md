# Multivariate exponential-kernel Hawkes: log-likelihood

Status: **awaiting owner approval.** Per CLAUDE.md §4 no implementation may proceed
until this is approved.

Scope: `d` components with cross-excitation, exponential kernel, one observation window
`[0, T]`. Generalises `univariate_loglikelihood.md`, whose equation numbers `(4.4)`,
`(4.5)` are referenced throughout.

## 0. Citations

| Symbol / step | Source |
| --- | --- |
| mutually exciting Hawkes process | [Laub2015, Definition 7, eq. 12] |
| exponential form, strict bounds, index orientation | [Laub2015, eq. 13] |
| stationarity condition | [Bacry2015, Proposition 1] |
| stationary mean intensity | [Bacry2015, Proposition 4, eq. 21] with eq. 17 |
| kernel normalization | `conventions.md` C1, experiments E1, E2 |
| predictable intensity, strict bounds | `conventions.md` C3, experiment E2b |
| compensator integrates to `T` | `conventions.md` C4 |
| `T` supplied by the caller | `conventions.md` C5 |
| `alpha[i][j]` means "j excites i" | `conventions.md` C6, and [Laub2015, eq. 13] |
| ties, ordering, input contract | `conventions.md` C8, experiment E3 |

[Bacry2015] is Bacry, Mastromatteo & Muzy, *Hawkes processes in finance*,
arXiv:1502.04592, freely accessible. Its PDF is not committed
(`docs/references/README.md`).

### 0.1 [Laub2015] is in a different parametrization, again

[Laub2015, eq. 13] writes

```
lambda*_i(t) = lambda_i + sum_{j=1}^{m} sum_{t_{j,k} < t} alpha_L[i][j] * exp(-beta[i][j]*(t - t_{j,k}))
```

Two differences from what is coded, both carried forward from M1 §1.1:

- **Kernel normalization.** `hawk` uses `alpha[i][j] * beta * exp(-beta t)`
  (`conventions.md` C1), so `alpha_L[i][j] = alpha[i][j] * beta`.
- **A single decay.** [Laub2015] allows a full matrix `beta[i][j]`. `hawk` uses one
  scalar `beta` shared by every pair, matching `tick`'s
  `ModelHawkesExpKernLogLik(decay: float)`, which takes a scalar and is the differential
  oracle. A per-pair decay matrix is **out of scope**: nothing in the corpus or in
  `tick`'s likelihood estimator can check it, and CLAUDE.md §5 forbids generalizing
  ahead of a caller.

### 0.2 What [Laub2015, eq. 13] independently confirms

Until now the index orientation rested on `tick`'s source alone (`conventions.md` C6).
Eq. 13 sums `alpha_L[i][j]` against the events of component `j` inside the intensity of
component `i`, so `alpha[i][j]` means **"j excites i"** — the same orientation, from the
primary reference. Eq. 13 also writes the sum as `t_{j,k} < t`, strictly, confirming C3
in the multivariate case.

## 1. Setup and notation

| Symbol | Meaning |
| --- | --- |
| `d` | number of components |
| `mu[i] > 0` | baseline intensity of component `i` |
| `alpha[i][j] >= 0` | branching ratio from `j` into `i`; **j excites i** |
| `beta > 0` | exponential decay rate, shared by all pairs (§0.1) |
| `T` | observation horizon, supplied by the caller (C5) |
| `t^j_k` | the `k`-th event of component `j` |
| `n_j` | number of events of component `j`; `n = sum_j n_j` |

Events within each component are ascending, all in `[0, T]`, ties permitted (C8).

**The orientation is load-bearing and appears in every expression below.** The first
index of `alpha` is always the component being excited, the second the component doing
the exciting. A transposed matrix produces plausible numbers and is only detectable on
asymmetric data.

## 2. Intensity and compensator

**Intensity** (C3, C6; [Laub2015, eq. 13] under §0.1's substitution):

```
lambda_i(t) = mu[i] + sum_{j=1}^{d} sum_{t^j_k < t} alpha[i][j] * beta * exp(-beta*(t - t^j_k))
```

The inner sum is over events of component `j` **strictly** before `t`. Three
consequences, each of which an implementation can get wrong:

- `lambda_i(t)` at the earliest event in the whole system equals `mu[i]`.
- An event of component `j` at exactly time `t` does **not** contribute to
  `lambda_i(t)`, for any `i`, **including `i = j` and including `i != j`**. This is the
  cross-component tie case; see §4.1.
- The condition is on *times*, never on positions in an array.

**Compensator.** Integrating over `[0, T]`, exactly as in M1 §2 but once per
`(i, j)` pair:

```
Lambda_i(T) = int_0^T lambda_i(u) du
            = mu[i]*T + sum_{j=1}^{d} alpha[i][j] * sum_{k=1}^{n_j} ( 1 - exp(-beta*(T - t^j_k)) )
                                                                            (M2.1)
```

Note the inner sum depends only on `j`, not on `i`. Define

```
E_j := sum_{k=1}^{n_j} ( 1 - exp(-beta*(T - t^j_k)) )                       (M2.2)
```

so `Lambda_i(T) = mu[i]*T + sum_j alpha[i][j] * E_j`. There are `d` such sums, not
`d^2`.

## 3. Log-likelihood

For a `d`-variate point process the log-likelihood is the sum of the per-component
contributions ([Laub2015, Theorem 3] applied to each component, with the intensities
coupled through §2):

```
nll = sum_{i=1}^{d} [ Lambda_i(T) - sum_{k=1}^{n_i} log lambda_i(t^i_k) ]     (M3.1)
```

Written out:

```
nll = sum_i mu[i]*T
    + sum_i sum_j alpha[i][j] * E_j
    - sum_i sum_{k=1}^{n_i} log( mu[i] + sum_j sum_{t^j_l < t^i_k} alpha[i][j]*beta*exp(-beta*(t^i_k - t^j_l)) )
                                                                            (M3.2)
```

(M3.2) is the **definition**. Its log term is `O(n^2)` and its compensator `O(n*d)`.
M2 Part B step 6 transcribes exactly this, with no simplification, as the reference
oracle.

`lambda_i(t) >= mu[i] > 0` always, so the logarithm is never evaluated at zero or below.

### 3.1 With ties, this is not a likelihood

The M1 caveat (§3.1 there) carries over unchanged and now has a second face: ties may
occur **within** a component or **across** components, and in both cases
[Laub2015, Theorem 3]'s simple-point-process assumption fails. The expression still
evaluates, the intensity path is still well defined, and it still agrees with `tick`,
but it is not the density of anything the model can generate. The maximum-likelihood
asymptotics do not apply, and the round-trip property test must not synthesise ties.

## 4. The grouped recursion

Define, for each component `m`, the **excitation state**

```
A_m(t) := sum_{t^m_k < t} exp(-beta*(t - t^m_k))                            (M4.1)
```

so that, factoring `beta` and `alpha[i][j]` out of the inner sum,

```
lambda_i(t) = mu[i] + sum_{j=1}^{d} alpha[i][j] * beta * A_j(t)             (M4.2)
```

There are `d` states, one per component, and every intensity is a weighted sum of all
of them. This is the whole structural difference from M1, where `d = 1` and there is a
single state.

### 4.1 Grouping must pool ALL components

M1 §4.2 established that the recursion must advance by **distinct time**, not by event,
or a tie is counted as exciting itself. In `d` dimensions the same requirement holds
across components, and the failure is easier to miss because the two events sit in
different arrays.

Consider walking the pooled event stream and advancing the states at every *event*.
When two components share a timestamp, the event processed first is added to its state
before the second event's intensity is evaluated — so it excites it, which §2 forbids.
Whether the answer comes out right then depends on which array the merge happened to
visit first, which is not a property the mathematics has.

**Counterexample.** `d = 2`, `mu = [0.2, 0.5]`,
`alpha = [[0.1, 0.6], [0.05, 0.15]]`, `beta = 1.2`, `T = 6`, with

```
component 0:  [1.0, 2.5]
component 1:  [2.5, 4.0]          <- 2.5 is shared
```

| expression | value |
| --- | --- |
| definition (M3.2) | 10.329672183654557 |
| grouped over pooled distinct times | 10.329672183654555 |
| advanced per event | **10.218429607528986** |

The per-event walk is wrong by about 1.1%. Move component 1's first event from `2.5` to
`2.6` and the tie disappears; all three then agree at `10.223981924928560`. So the
defect is invisible on any data without a shared timestamp, which is to say invisible on
anything the simulator produces (M1: Ogata thinning yields ties with probability zero)
and on five of the ten committed fixtures.

Reproduced by `docs/derivations/check_multivariate_derivation.py`.

### 4.2 The recursion

Let

```
s_1 < s_2 < ... < s_M      the distinct event times, POOLED over all components
c^m_r                      the number of events of component m at time s_r
                           (possibly 0; sum_r c^m_r = n_m)
d_r := s_r - s_{r-1} > 0
```

and let `B^m_r := A_m(s_r)`. Because every intensity depends on `t` only through the
states, all events at `s_r` — whichever components they belong to — see the same state
vector.

**Base case.** No event lies strictly before the earliest event time:

```
B^m_1 = 0            for every m = 1 .. d                                   (M4.3)
```

**Step.** For `r >= 2` and each `m`, the events strictly before `s_r` are those strictly
before `s_{r-1}` plus the `c^m_{r-1}` events of component `m` at `s_{r-1}`:

```
B^m_r = exp(-beta*d_r) * ( B^m_{r-1} + c^m_{r-1} )                          (M4.4)
```

which is (4.4) applied independently to each component, with the **same** `d_r` for all
of them — that shared `d_r` is what pooling buys, and what per-component grouping would
lose.

**Intensity.** For any component `i` with `c^i_r > 0`:

```
lambda_i(s_r) = mu[i] + sum_{j=1}^{d} alpha[i][j] * beta * B^j_r            (M4.5)
```

**Assembled.**

```
nll = sum_i mu[i]*T
    + sum_i sum_j alpha[i][j] * E_j
    - sum_r sum_i c^i_r * log( lambda_i(s_r) )                              (M4.6)
```

with `E_j = sum_r c^j_r * (1 - exp(-beta*(T - s_r)))`.

**Cost.** `O(M*d)` for the state recursion and `O(n*d)` for the intensities, so
`O(n*d)` overall, against `O(n^2)` for (M3.2).

## 5. The exact expression to be coded

Inputs: `events[0..d-1]`, each ascending and within `[0, T]`; `mu[0..d-1]`,
`alpha[0..d-1][0..d-1]`, `beta`, all positive (`alpha` non-negative).

```
if every component is empty:
    return sum_i mu[i]*T

s, counts = distinct times pooled across ALL components, ascending,
            with counts[r][m] = number of events of component m at s[r]

state[m]        = 0.0  for m in 0..d-1               # B^m_r,  (M4.3)
compensator[j]  = 0.0  for j in 0..d-1               # E_j
log_term        = 0.0

for r in 0 .. len(s)-1:
    if r > 0:
        gap   = s[r] - s[r-1]
        decay = exp(-beta * gap)
        for m in 0 .. d-1:
            state[m] = decay * (state[m] + counts[r-1][m])      # (M4.4)

    window = T - s[r]
    contribution = -expm1(-beta * window)
    for j in 0 .. d-1:
        repeat counts[r][j] times:                              # NOT counts * value
            compensator[j] += contribution

    for i in 0 .. d-1:
        if counts[r][i] == 0: continue
        intensity = mu[i]
        for j in 0 .. d-1:
            intensity += (alpha[i][j] * beta) * state[j]        # (M4.5)
        ln_intensity = ln(intensity)
        repeat counts[r][i] times:                              # NOT counts * value
            log_term += ln_intensity

nll = sum_i mu[i]*T
for i in 0 .. d-1:
    for j in 0 .. d-1:
        nll += alpha[i][j] * compensator[j]
return nll - log_term
```

### 5.1 Two accumulation rules that are not stylistic

Both exist so that §6's `d = 1` reduction is **bitwise** exact, which M2 Part B step 9
asserts. Neither is a numerical improvement; they are compatibility constraints.

- **Accumulate per event, not `count * value`.** For a tie of multiplicity `c`,
  `c * x` and `x` added `c` times are not the same `f64`. They agree for `c` in
  {1, 2, 3, 5} for the value tested and differ at `c = 7`:
  `7*x = 4.182330431764548` against `4.182330431764549` accumulated. M1 accumulates per
  event, so this must too.
- **Group the intensity product as `(alpha[i][j] * beta) * state[j]`.** Floating-point
  multiplication is not associative, and M1 computes `(alpha*beta)*B`. Writing
  `beta * (alpha[i][j] * state[j])` or `alpha[i][j] * (beta * state[j])` is
  mathematically identical and bitwise not.

## 6. Reduction to `d = 1`

Set `d = 1`, write `mu = mu[0]`, `alpha = alpha[0][0]`, and drop the component indices.

- **States (M4.3), (M4.4).** One state, `B_r = exp(-beta*d_r)*(B_{r-1} + c_{r-1})` with
  `B_1 = 0`. This is **(4.4)** exactly, and the pooled distinct times are the distinct
  times of the single component, so §4.1's pooling is vacuous.
- **Intensity (M4.5).** The sum over `j` has one term:
  `lambda(s_r) = mu + (alpha*beta)*B_r`. This is M1's `intensity_at`.
- **Compensator.** One `E_1 = sum_r c_r*(1 - exp(-beta*(T - s_r)))`, which is M1's
  `compensator_excitation`.
- **Assembly (M4.6).** `mu*T + alpha*E_1 - log_term`, which is **(4.5)**.

So (M4.6) collapses symbolically to (4.5). With §5.1's two rules it collapses
**bitwise** as well: verified on 600 randomized cases, half of them tie-heavy, all 600
exact.

## 7. Stationary mean intensity

The only oracle anchored outside the implementation, and the multivariate counterpart
of [Laub2015, eq. 6].

Suppose the process is stationary and write `Lambda_i = E[lambda_i(t)]`, constant in
`t`. Taking expectations in §2's intensity, and using `E[dN_j(u)] = Lambda_j du`:

```
Lambda_i = mu[i] + sum_j alpha[i][j] * beta * int_0^inf exp(-beta*s) * Lambda_j ds
         = mu[i] + sum_j alpha[i][j] * beta * Lambda_j * (1/beta)
         = mu[i] + sum_j alpha[i][j] * Lambda_j
```

The `beta` cancels, as it must: the kernel integrates to `alpha[i][j]` regardless of
`beta` (C1, C2). In matrix form, with `alpha` the branching matrix:

```
Lambda = mu + alpha * Lambda        =>     ( I - alpha ) * Lambda = mu

Lambda = ( I - alpha )^{-1} * mu                                            (M7.1)
```

**Invertibility condition.** [Bacry2015, Proposition 1] requires the spectral radius of
the matrix of kernel `L1` norms to be `< 1`; under C1 that matrix is `alpha` itself, so

```
spectral_radius(alpha) < 1                                                  (M7.2)
```

Under (M7.2), `I - alpha` is invertible, the Neumann series
`(I - alpha)^{-1} = sum_{k>=0} alpha^k` converges, and `Lambda >= mu > 0` componentwise
for non-negative `alpha`. (M7.2) is strictly stronger than invertibility of `I - alpha`:
that matrix can be invertible with `spectral_radius(alpha) > 1`, in which case (M7.1)
returns a vector that solves the linear system and is not a mean intensity — it may even
have negative entries. **The condition must be checked before (M7.1) is used**, not
inferred from the solve succeeding.

[Bacry2015, Proposition 4, eq. 21] states this as `Lambda = (I + Psi_hat(0)) mu`, where
`Psi` is the inverted kernel of its eq. 17, `Psi = Phi + Psi * Phi`. Taking Laplace
transforms at zero, `Psi_hat(0) = Phi_hat(0) + Psi_hat(0) Phi_hat(0)`, so
`Psi_hat(0) = Phi_hat(0)(I - Phi_hat(0))^{-1}` and
`I + Psi_hat(0) = (I - Phi_hat(0))^{-1}`. With `Phi_hat(0) = alpha`, that is (M7.1).

**At `d = 1`:** `(1 - alpha)^{-1} * mu = mu/(1 - alpha)`, which is [Laub2015, eq. 6] and
M1's `stationary_mean_intensity`. Checked: `mu = 0.6`, `alpha = 0.5` gives `1.2` from
both.

**Worked example**, used as a test vector. `mu = [0.2, 0.5]`,
`alpha = [[0.1, 0.6], [0.05, 0.15]]`. Eigenvalues of `alpha` are `0.3` and `-0.05`, so
`spectral_radius = 0.3 < 1`. `I - alpha = [[0.9, -0.6], [-0.05, 0.85]]`,
`det = 0.735`, and

```
Lambda = (1/0.735) * [[0.85, 0.6], [0.05, 0.9]] * [0.2, 0.5]
       = (1/0.735) * [0.47, 0.46]
       = [0.6394557823129251, 0.6258503401360543]
```

## 8. Edge cases

| Case | Behaviour |
| --- | --- |
| every component empty | `nll = sum_i mu[i]*T`. No log term. |
| some components empty | Those contribute `mu[i]*T` and no log terms; their `E_j = 0`, so they excite nothing. Others are unaffected. |
| `d = 1` | §6. Reduces to M1 bitwise. |
| tie within one component | Neither event excites the other (C3, C8). |
| tie across components | Neither excites the other, in either direction. §4.1. |
| `alpha` with a zero row | Component `i` is a plain Poisson process; the expression handles it with no special case. |
| `alpha` with a zero column | Component `j` excites nothing. Its own events still appear in its log term. |
| `t = 0` or `t = T` present | Allowed (C8). An event at `T` contributes `1 - exp(0) = 0` to the compensator. |
| unsorted input within a component | Rejected as an error (C8). |

## 9. Numerical check run during Part A

`docs/derivations/check_multivariate_derivation.py` — throwaway Python, no
dependencies, **not** `hawk` code.

```
1. recursion (M4.6) vs definition (M3.2), worst relative : 2.122e-14
   600 cases, d in 1..5, half tie-heavy
3. d=1 vs M1 univariate nll, bitwise                     : 600/600 exact
   d=1 vs M1 univariate gradient, bitwise                : 600/600 exact
4. gradient vs central differences, worst relative       : 4.233e-08
5. stationary mean at d=1                                : 1.2 = mu/(1-alpha)
```

The `d = 1` bitwise results hold against M1 **as shipped**, including the association
`alpha * (X - Y)` its `d_beta` uses. That was not automatic — see
`multivariate_gradient.md` §4.3.

This is a consistency check, not verification: it is Python, it shares an author with
the derivation, and it is not sabotage-tested. Part B redoes all of it in Rust through
the M0/M1 harnesses.
