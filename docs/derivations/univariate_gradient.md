# Univariate exponential-kernel Hawkes: analytic gradient

Status: **awaiting owner approval.** Per CLAUDE.md §4 no implementation may proceed
until this is approved.

Companion to `univariate_loglikelihood.md`. All notation, conventions and citations
are inherited from it; equation numbers of the form `(4.4)` refer to that document.

A wrong derivative still converges — to the wrong place. Oracle 4 (finite
differences) is the only thing that catches it, which is why every partial below is
written out rather than sketched.

## 1. What is being differentiated

From (4.5), with `s_1 < ... < s_m` the distinct event times, `c_j` their
multiplicities, and `B_j` the excitation state (4.3)-(4.4):

```
nll(mu, alpha, beta)
    = mu*T
    + alpha * sum_{j=1}^{m} c_j * ( 1 - exp(-beta*(T - s_j)) )
    - sum_{j=1}^{m} c_j * log( lambda_j )

lambda_j = mu + alpha * beta * B_j(beta)
```

The one subtlety in the whole document: **`B_j` depends on `beta`.** It is a
function of the data and `beta` only — not of `mu`, not of `alpha`. So `B_j` is a
constant for the first two partials and needs its own recursion for the third.

Write:

```
E_j := 1 - exp(-beta*(T - s_j))              the compensator factor at s_j
W_j := T - s_j >= 0                          the remaining window after s_j
Bp_j := d B_j / d beta                       the state derivative
```

## 2. Partial with respect to `mu`

`lambda_j` depends on `mu` through the additive term only, so
`d lambda_j / d mu = 1`:

```
d nll / d mu = T - sum_{j=1}^{m} c_j / lambda_j                             (G.1)
```

**Independently confirmed.** Experiment E2 in
`benchmarks/docker/convention_experiments.py` predicts `tick`'s
`d(loss*n_jumps)/dmu` from `T - sum_k 1/lambda(t_k)` and matches
`-0.6399003563316408` to 1e-9. `tick` normalizes by `n_jumps`; the shape is (G.1).

## 3. Partial with respect to `alpha`

The compensator term is linear in `alpha`. In the log term,
`d lambda_j / d alpha = beta * B_j`:

```
d nll / d alpha = sum_{j=1}^{m} c_j * E_j
                - sum_{j=1}^{m} c_j * beta * B_j / lambda_j                 (G.2)
```

**Independently confirmed.** Experiment E2 predicts `tick`'s
`d(loss*n_jumps)/dalpha` as `0.8395209932253471` from this shape and matches to
1e-9. The competing kernel convention predicts `0.5843061858675322`, so this
partial is also a check on C1.

## 4. Partial with respect to `beta`

Two contributions, and this is where an error is easiest to hide.

### 4.1 Compensator term

```
d/d beta [ alpha * sum_j c_j * ( 1 - exp(-beta*W_j) ) ]
    = alpha * sum_j c_j * W_j * exp(-beta*W_j)                              (G.3)
```

using `d/d beta [ -exp(-beta*W) ] = W*exp(-beta*W)`. Non-negative, as it must be:
increasing `beta` shortens the kernel's memory, so more of each kernel's mass falls
inside the window.

### 4.2 Log term

By the product rule on `lambda_j = mu + alpha*beta*B_j(beta)`:

```
d lambda_j / d beta = alpha * ( B_j + beta * Bp_j )                         (G.4)
```

**Dropping `beta * Bp_j` is the error this document exists to prevent.** It leaves a
derivative that is wrong but plausible: right sign, right order of magnitude,
converging to the wrong optimum.

### 4.3 The state derivative `Bp_j`

Differentiate the recursion (4.4) with respect to `beta`. Base case, from
`B_1 = 0` (4.3) — a constant, independent of `beta`:

```
Bp_1 = 0                                                                    (G.5)
```

Step, for `j >= 2`, with `d_j = s_j - s_{j-1} > 0`. Note `d_j` and `c_{j-1}` do not
depend on `beta`:

```
B_j = exp(-beta*d_j) * ( B_{j-1} + c_{j-1} )

Bp_j = d/d beta [ exp(-beta*d_j) ] * ( B_{j-1} + c_{j-1} )
     + exp(-beta*d_j) * Bp_{j-1}

     = -d_j * exp(-beta*d_j) * ( B_{j-1} + c_{j-1} )
     + exp(-beta*d_j) * Bp_{j-1}

     = -d_j * B_j + exp(-beta*d_j) * Bp_{j-1}                               (G.6)
```

The substitution in the last line reuses `B_j` from (4.4), so `Bp_j` costs one
multiply and one add beyond the value recursion. Both are carried in the same pass.

`Bp_j <= 0` throughout: raising `beta` can only shrink accumulated excitation.

### 4.4 Assembled

```
d nll / d beta = alpha * sum_{j=1}^{m} c_j * W_j * exp(-beta*W_j)
               - alpha * sum_{j=1}^{m} c_j * ( B_j + beta*Bp_j ) / lambda_j (G.7)
```

**`tick` cannot check this one.** `ModelHawkesExpKernLogLik` takes `decay` as a
fixed constructor argument, not as a coefficient, so it exposes no derivative with
respect to `beta`. (G.1) and (G.2) were confirmed against `tick`; (G.7) rests on
this derivation plus oracle 4. That asymmetry is the reason step 8's finite-
difference check is a gate and not a formality.

## 5. The exact expression to be coded

One pass, `O(n)`, computing `nll` and all three partials together — the excitation
state is needed by all of them.

```
if n == 0:
    return nll = mu*T,  d_mu = T,  d_alpha = 0,  d_beta = 0

compensator_excitation = 0.0     # sum_j c_j * E_j
log_term               = 0.0     # sum_j c_j * log(lambda_j)
d_mu_accumulator       = 0.0     # sum_j c_j / lambda_j
d_alpha_accumulator    = 0.0     # sum_j c_j * beta * B_j / lambda_j
d_beta_compensator     = 0.0     # sum_j c_j * W_j * exp(-beta*W_j)
d_beta_log             = 0.0     # sum_j c_j * (B_j + beta*Bp_j) / lambda_j

B  = 0.0                         # B_j        (4.3)
Bp = 0.0                         # Bp_j       (G.5)
previous_time  = t[0]
previous_count = 0

for k in 0 .. n-1:
    if t[k] != previous_time:                        # new distinct time s_j
        d     = t[k] - previous_time
        decay = exp(-beta * d)
        B_new = decay * (B + previous_count)         # (4.4)
        Bp    = -d * B_new + decay * Bp              # (G.6)  uses B_new, not B
        B     = B_new
        previous_time  = t[k]
        previous_count = 0

    intensity = mu + alpha * beta * B                # lambda_j
    window    = T - t[k]                             # W_j
    decay_w   = exp(-beta * window)

    log_term               += ln(intensity)
    compensator_excitation += -expm1(-beta * window)
    d_mu_accumulator       += 1.0 / intensity
    d_alpha_accumulator    += beta * B / intensity
    d_beta_compensator     += window * decay_w
    d_beta_log             += (B + beta * Bp) / intensity

    previous_count += 1

nll     = mu*T + alpha * compensator_excitation - log_term
d_mu    = T - d_mu_accumulator                                    # (G.1)
d_alpha = compensator_excitation - d_alpha_accumulator            # (G.2)
d_beta  = alpha * (d_beta_compensator - d_beta_log)               # (G.7)
```

Two ordering hazards, both of which produce a subtly wrong `d_beta`:

1. `Bp` must be updated **using the new `B`** — (G.6) is `-d_j*B_j + ...`, with
   `B_j` the post-update value. Computing `Bp` from the old `B` is wrong.
2. `Bp` must be updated **before** `B` is overwritten, or written via a temporary as
   above.

## 6. Log-parameter space

Positivity of `mu, alpha, beta` is handled by optimizing over
`theta = (ln mu, ln alpha, ln beta)` rather than by constrained optimization
(CLAUDE.md §6). Conversion happens at the boundary, by the chain rule
`d nll / d ln x = x * d nll / dx`:

```
d nll / d ln mu    = mu    * d nll / d mu
d nll / d ln alpha = alpha * d nll / d alpha
d nll / d ln beta  = beta  * d nll / d beta                                 (G.8)
```

The finite-difference check (Part B step 8) must be run in **both**
parametrizations. (G.8) is where a factor can silently go missing, and a check only
in natural parameters would not see it.

## 7. Verification plan

| Check | Covers |
| --- | --- |
| Central differences vs (G.1), (G.2), (G.7) at randomized parameter points, to 1e-6 | all three partials, especially `d_beta` |
| Central differences in log space vs (G.8) | the chain-rule conversion |
| (G.1), (G.2) vs `tick`'s gradient | independent confirmation of two of three |
| Sabotage: drop `beta*Bp_j` from (G.4) | the term most likely to be omitted |
| Sabotage: compute `Bp_j` from the pre-update `B` | hazard 1 of §5 |
| `n == 0` and `n == 1` | base cases |

The `1e-6` tolerance is the M0 gradient harness's, justified there from the central
difference's own round-off floor. It is not tuned to make anything pass.

## 8. Numerical check run during Part A

Checked before handover by `docs/derivations/check_univariate_derivation.py`, the
same script described in `univariate_loglikelihood.md` §8. Over 454 cases including
tie-heavy and degenerate ones, the worst relative disagreement between (G.1), (G.2),
(G.7) and central differences was

```
1.828e-08
```

comfortably inside the `1e-6` gate, and consistent with the central difference's own
round-off floor rather than with a systematic error.

This is a consistency check, not verification: it is Python, it shares an author with
the derivation, and it has not been sabotage-tested. Part B step 8 redoes it in Rust
through the M0 gradient harness, which has been.
