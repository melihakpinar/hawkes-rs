# Multivariate exponential-kernel Hawkes: analytic gradient

Status: **awaiting owner approval.** Per CLAUDE.md §4 no implementation may proceed
until this is approved.

Companion to `multivariate_loglikelihood.md`; notation, conventions and citations are
inherited from it, and `(M4.4)`, `(M4.6)` refer there. `(G.1)`-`(G.8)` refer to
`univariate_gradient.md`.

There are `d + d^2 + 1` parameters: `mu[i]`, `alpha[i][j]`, and the shared `beta`.

## 1. What is being differentiated

From (M4.6):

```
nll = sum_i mu[i]*T
    + sum_i sum_j alpha[i][j] * E_j
    - sum_r sum_i c^i_r * log( lambda_i(s_r) )

lambda_i(s_r) = mu[i] + sum_j alpha[i][j] * beta * B^j_r
E_j           = sum_r c^j_r * ( 1 - exp(-beta*(T - s_r)) )
```

As in M1, the one subtlety is that `B^j_r` depends on `beta` — and now there are `d` of
them. `B^j_r` is a function of the data and `beta` only; it does not depend on `mu` or
on `alpha`. Write

```
W_r   := T - s_r >= 0
Bp^j_r := d B^j_r / d beta
```

## 2. Partial with respect to `mu[i]`

Only component `i`'s own log terms involve `mu[i]`, and
`d lambda_i / d mu[i] = 1`:

```
d nll / d mu[i] = T - sum_r c^i_r / lambda_i(s_r)                           (MG.1)
```

`d` partials, each touching only its own component. At `d = 1` this is (G.1).

## 3. Partial with respect to `alpha[i][j]`

`alpha[i][j]` appears twice: in the compensator, linearly, against `E_j`; and in
`lambda_i`, where `d lambda_i / d alpha[i][j] = beta * B^j_r`. It appears in **no other
component's** intensity — that is what the orientation "j excites i" means, and getting
it wrong here transposes the fitted matrix.

```
d nll / d alpha[i][j] = E_j - sum_r c^i_r * ( beta * B^j_r ) / lambda_i(s_r)  (MG.2)
```

Note carefully which index goes where: the compensator term is `E_j`, indexed by the
**exciting** component; the sum is over the events of the **excited** component `i`,
weighted by that component's intensity; and the state is `B^j_r`, again the exciting
one. All three would survive a transposition on symmetric data and none would on
asymmetric data.

At `d = 1` this is (G.2).

## 4. Partial with respect to `beta`

### 4.1 Compensator term

```
d E_j / d beta = sum_r c^j_r * W_r * exp(-beta*W_r)                         (MG.3)
```

using `d/d beta [ -exp(-beta*W) ] = W*exp(-beta*W)`, exactly as (G.3). Non-negative.

### 4.2 Log term

By the product rule on `lambda_i(s_r) = mu[i] + sum_j alpha[i][j]*beta*B^j_r`:

```
d lambda_i(s_r) / d beta = sum_j alpha[i][j] * ( B^j_r + beta * Bp^j_r )    (MG.4)
```

**The term (G.4) warns about now appears once per `(i, j)` pair.** In M1 there was a
single `beta * Bp` that could be dropped; here there are `d^2` of them, and dropping any
subset leaves a derivative that is wrong only in the directions involving those pairs.
On a symmetric `alpha` with symmetric data such a bug can hide almost completely.

### 4.3 The state-derivative recursion

Differentiate (M4.4) with respect to `beta`, per component. `d_r` and `c^m_{r-1}` do not
depend on `beta`:

```
Bp^m_1 = 0                                                                  (MG.5)

Bp^m_r = -d_r * B^m_r + exp(-beta*d_r) * Bp^m_{r-1}                         (MG.6)
```

identical in form to (G.6), run `d` times with a shared `d_r`. `B^m_r` here is the
**post-update** value from (M4.4); computing it from the pre-update state is the same
hazard as M1's, now available in `d` copies.

`Bp^m_r <= 0` throughout.

### 4.4 Assembled, and the accumulation that makes `d = 1` bitwise

The obvious assembly is

```
d nll / d beta = sum_i sum_j alpha[i][j] * dE_j
               - sum_r sum_i c^i_r * ( sum_j alpha[i][j]*(B^j_r + beta*Bp^j_r) ) / lambda_i(s_r)
```

That is correct, and it does **not** reduce bitwise to M1, which computes
`alpha * (X - Y)` with both `X` and `Y` accumulated first. Factoring differently fixes
it. Define, per pair,

```
S[i][j] := sum_r c^i_r * ( B^j_r + beta * Bp^j_r ) / lambda_i(s_r)          (MG.7)
```

— note the sum runs over the events of the excited component `i`, while the state is
that of the exciting component `j` — and then

```
d nll / d beta = sum_i sum_j alpha[i][j] * ( dE_j - S[i][j] )               (MG.8)
```

(MG.8) equals the expression above by distributing `alpha[i][j]`, and at `d = 1` it is
`alpha * (dE - S)`, which is exactly the association M1's shipped code uses. Verified:
600 randomized cases, `d = 1`, gradient bitwise identical to M1 in all 600.

The alternative factoring requires re-associating M1's `d_beta`, i.e. amending code and
a derivation that have already been approved. (MG.8) avoids that, at the cost of one
`d x d` accumulator — the same shape as the one (MG.2) already needs.

`tick` cannot check any of this: `decay` is a fixed constructor argument there, so it
exposes no derivative with respect to `beta` at any `d`.

## 5. The exact expression to be coded

One pass, `O(n*d + n*d^2)`; the `d^2` term is the pair accumulators, not the data.

```
state[m]      = 0.0            # B^m_r        (M4.3)
state_deriv[m]= 0.0            # Bp^m_r       (MG.5)
compensator[j]= 0.0            # E_j
dcompensator[j]= 0.0           # dE_j
log_term      = 0.0
acc_mu[i]     = 0.0            # sum_r c^i_r / lambda_i
acc_alpha[i][j]= 0.0           # sum_r c^i_r * beta*B^j_r / lambda_i
S[i][j]       = 0.0            # (MG.7)

for r in 0 .. M-1:
    if r > 0:
        gap   = s[r] - s[r-1]
        decay = exp(-beta * gap)
        for m in 0 .. d-1:
            advanced        = decay * (state[m] + counts[r-1][m])   # (M4.4)
            state_deriv[m]  = -gap * advanced + decay * state_deriv[m]   # (MG.6)
            state[m]        = advanced
            # state_deriv uses `advanced`, and is written before `state[m]` is
            # overwritten. Both hazards are M1 §5's, once per component.

    window = T - s[r]
    window_decay = exp(-beta * window)
    contribution = -expm1(-beta * window)
    for j in 0 .. d-1:
        repeat counts[r][j] times:
            compensator[j]  += contribution
            dcompensator[j] += window * window_decay

    for i in 0 .. d-1:
        if counts[r][i] == 0: continue
        intensity = mu[i]
        for j in 0 .. d-1:
            intensity += (alpha[i][j] * beta) * state[j]
        ln_intensity = ln(intensity)
        repeat counts[r][i] times:
            log_term   += ln_intensity
            acc_mu[i]  += 1.0 / intensity
            for j in 0 .. d-1:
                acc_alpha[i][j] += (beta * state[j]) / intensity
                S[i][j]         += (state[j] + beta * state_deriv[j]) / intensity

nll = sum_i mu[i]*T
for i, j: nll += alpha[i][j] * compensator[j]
nll -= log_term

d_mu[i]       = T - acc_mu[i]                                       # (MG.1)
d_alpha[i][j] = compensator[j] - acc_alpha[i][j]                    # (MG.2)
d_beta        = sum_i sum_j alpha[i][j] * (dcompensator[j] - S[i][j])   # (MG.8)
```

The `repeat counts times` loops and the `(alpha*beta)*state` grouping are the same
compatibility rules as `multivariate_loglikelihood.md` §5.1, and they apply to the
gradient accumulators for the same reason.

## 6. Log-parameter space

Positivity is handled by optimizing over `ln mu[i]`, `ln alpha[i][j]`, `ln beta`
(CLAUDE.md §6). By the chain rule, exactly as (G.8):

```
d nll / d ln mu[i]       = mu[i]       * d nll / d mu[i]
d nll / d ln alpha[i][j] = alpha[i][j] * d nll / d alpha[i][j]
d nll / d ln beta        = beta        * d nll / d beta                     (MG.9)
```

`alpha[i][j] = 0` is a legitimate parameter value and `ln 0` is not. Whether the fit
optimizes in log space over `alpha` — which excludes exact zeros and makes a zero
entry reachable only in the limit — or handles `alpha` differently, is an
**implementation decision for Part B step 13**, not settled here. It did not arise in
M1 because a univariate `alpha` at zero is a degenerate Poisson process; in `d`
dimensions a sparse `alpha` with exact zeros is ordinary. Recorded so it is decided
deliberately rather than by whichever the first draft happens to do.

The finite-difference check must run in both parametrizations, per M1's experience with
(G.8).

## 7. Verification plan

| Check | Covers |
| --- | --- |
| Central differences vs (MG.1), (MG.2), (MG.8) over randomized `d` and parameters, 1e-6 | all partials, especially `d_beta` |
| Central differences in log space vs (MG.9) | the chain-rule conversion |
| `d = 1` bitwise against M1's gradient | that the generalisation did not perturb the base case |
| Asymmetric `alpha` with `alpha[i][j] != alpha[j][i]` | (MG.2)'s index orientation |
| Sabotage: transpose `alpha` in (MG.2) | the transposition a symmetric test cannot see |
| Sabotage: drop `beta*Bp^j` from (MG.4) for one `(i, j)` pair only | partial omission, which `d = 1` could not express |
| Sabotage: compute `Bp^m` from the pre-update state | M1 hazard 1, per component |
| Empty components, zero rows, zero columns of `alpha` | §8 of the likelihood derivation |

`1e-6` is the M0 gradient harness's tolerance, justified there. It is not tuned.

## 8. Numerical check run during Part A

`docs/derivations/check_multivariate_derivation.py`, described in
`multivariate_loglikelihood.md` §9:

```
gradient vs central differences, worst relative : 4.233e-08
d = 1 gradient vs M1 as shipped, bitwise        : 600/600 exact
```

Consistency check, not verification: Python, same author as the derivation, not
sabotage-tested. Part B redoes it in Rust through the M0 harness.
