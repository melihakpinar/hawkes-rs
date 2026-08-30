# Spectral radius of the branching matrix

Status: **awaiting owner approval**, and **retrospective**. This is not CLAUDE.md §4's normal order. The code in
`multivariate::Parameters::branching_ratio_spectral_radius` was written first, and two
bugs were found in it by tests before this document existed. It is written now because
the README audit found the routine to be the one formula in the repository with no
derivation behind it, and because the reasoning was scattered across a doc comment and
two test files with nothing tying it together.

Nothing here is new mathematics for the implementation. Every numeric claim below was
measured against the shipped routine through the Python bindings, and the measurement
is given rather than the assertion.

## 0. Citations

| Step | Source |
| --- | --- |
| stationarity requires `spectral_radius(alpha) < 1` | `conventions.md` C2; [Bacry2015, Proposition 1] |
| the branching matrix **is** `alpha` under this normalization | `conventions.md` C1, experiments E1, E2 |
| Collatz–Wielandt bounds | proved in full in §2 below |
| Perron–Frobenius: `rho(A)` is an eigenvalue of a non-negative `A` | Horn & Johnson, *Matrix Analysis* (provenance only, no equation number — see below) |
| Gelfand's formula `rho(A) = lim ||A^k||^(1/k)` | Horn & Johnson, *Matrix Analysis* (provenance only, no equation number — see below) |

No PDF for Horn & Johnson is in `docs/references/`, so per CLAUDE.md §2 those two are
cited for provenance **without equation numbers** — citing a numbered theorem from a book
this repository cannot open is exactly the fabricated-source failure §1 exists to
prevent.

§2's requirement is met a different way: the two results are the only external input, and
both are standard, so §2 below **proves what is actually used** from first principles. The
Collatz–Wielandt bounds are derived, not quoted. Nothing in the implementation rests on a
reference this repository cannot check.

## 1. What is being computed

`conventions.md` C2 establishes that under the `alpha*beta*exp(-beta*t)` normalization
the kernel integrates to `alpha_ij`, so the branching matrix is the excitation matrix
itself and the process is stationary exactly when

```text
rho(alpha) < 1
```

`alpha` is entrywise non-negative — `Error::InvalidExcitation` rejects anything else —
and that non-negativity is the only structure the method below needs. In particular
`alpha` is **not** assumed irreducible, symmetric, or diagonalizable, and all three
assumptions fail on matrices a caller can legitimately supply.

The value is a diagnostic, reported on a fit rather than enforced during it
(CLAUDE.md §6). What it is used *for* is deciding `rho < 1`; that is what fixes how much
accuracy is enough, in §6.

## 2. The Collatz–Wielandt bounds

Let `A >= 0` entrywise and `x > 0` strictly. Write

```text
lower(x) = min_i (A x)_i / x_i        upper(x) = max_i (A x)_i / x_i
```

**Claim.** `lower(x) <= rho(A) <= upper(x)` for every strictly positive `x`.

*Upper.* By definition of `upper(x)`, `A x <= upper(x) * x` componentwise. Because
`A >= 0`, multiplying a componentwise inequality by `A` preserves it, so by induction

```text
A^k x <= upper(x)^k * x        for all k >= 1
```

Let `m = min_i x_i > 0` and `M = max_i x_i`. For any `i, j`, and using `A^k >= 0`,

```text
(A^k)_ij * m  <=  (A^k)_ij * x_j  <=  (A^k x)_i  <=  upper(x)^k * x_i  <=  upper(x)^k * M
```

so every entry of `A^k` is at most `(M/m) * upper(x)^k`, giving
`||A^k||_inf <= d * (M/m) * upper(x)^k`. By Gelfand's formula,

```text
rho(A) = lim_k ||A^k||_inf^(1/k) <= lim_k (d * M/m)^(1/k) * upper(x) = upper(x)
```

since `(d*M/m)^(1/k) -> 1`. ∎

*Lower.* Symmetrically `A x >= lower(x) * x`, so `A^k x >= lower(x)^k * x`, and

```text
||A^k||_inf * M  >=  (A^k x)_i  >=  lower(x)^k * x_i  >=  lower(x)^k * m
```

giving `||A^k||_inf >= (m/M) * lower(x)^k` and `rho(A) >= lower(x)`. ∎

**Both bounds are valid at every iteration, for any strictly positive `x`.** No
convergence, no irreducibility, and no property of the particular `x` is needed. This is
what makes the bracket-closure exit safe: when `upper - lower` is small, `rho` is pinned
between two numbers that are each independently rigorous.

The two are *not* symmetric in their limiting behaviour, and §3 is about that.

## 3. Why the upper bound is returned, and not the midpoint

The Collatz–Wielandt characterisation is

```text
rho(A) = inf over x > 0 of upper(x)
```

for every non-negative `A`, while the matching

```text
rho(A) = sup over x > 0 of lower(x)
```

requires `A` to be **irreducible**. The implementation therefore returns `upper` and
discards `lower` (`let _ = lower;`), keeping `lower` only for the bracket-closure test.

### 3.1 The counterexample is a diagonal matrix

Take `A = diag(0.2, 0.7, 0.4)`, whose spectral radius is `0.7`. For a diagonal matrix
and *any* strictly positive `x`,

```text
(A x)_i / x_i = A_ii     exactly
```

so `lower(x) = 0.2` and `upper(x) = 0.7` for every `x`, at every iteration, forever. The
bracket is `[0.2, 0.7]` and never closes; there is no `x` that improves it. The upper
bound is already exact on the first step, and the midpoint is `0.45` — wrong in the
first significant figure, and wrong in the direction that reports a stationary process
as further from the boundary than it is.

This is not a contrived case. A diagonal `alpha` is a set of `d` independent univariate
Hawkes processes, which is a perfectly ordinary thing for a caller to fit.

Measured against the shipped routine: `0.700000000000`, error `0.0e+00`.

### 3.2 The implementation's `lower` is additionally corrupted, and it does not matter

Reducibility is also what lets a component of `x` decay to nothing. On the diagonal case
the first component shrinks by a factor `1.2/1.7` per iteration under the shifted
recursion of §4, and after about 2 400 steps it reaches the **subnormal** range and
sticks at `1.5e-323`. Subnormal quantization then makes the computed ratio
`(A x)_0 / x_0` read `1.3333...` instead of the exact `1.2`, so the routine's `lower`
drifts to `0.3333` and its midpoint at the iteration cap is `0.5167`, not `0.45`.

Both numbers are wrong and the conclusion is unchanged, but the distinction is worth
recording: `0.45` is the midpoint of the *mathematical* bracket, and `0.5167` is what the
code would actually return if it returned a midpoint. The doc comment on
`branching_ratio_spectral_radius` states `0.45`; that is the mathematical figure, not
the computed one.

The returned value is unaffected — `upper` reads `1.7` exactly at every step, including
after the underflow — because `upper` is a max over ratios and the corrupted component's
ratio is far below it. This is the second reason to prefer `upper`: it is the bound that
is robust to a component of the iterate vanishing, and vanishing components are exactly
what reducible matrices produce.

## 4. Why the iteration runs on `A + I`

**Claim.** For `A >= 0`, `rho(A + I) = rho(A) + 1`.

*Proof.* The eigenvalues of `A + I` are `lambda_i + 1` for the eigenvalues `lambda_i` of
`A`. By Perron–Frobenius, `r = rho(A)` is itself an eigenvalue of the non-negative `A`,
and it is real and non-negative, so `r + 1` is an eigenvalue of `A + I`. For any other
eigenvalue, `|lambda + 1| <= |lambda| + 1 <= r + 1`. Hence the maximum modulus is exactly
`r + 1`. ∎

So the shift is exactly invertible: iterate on `A + I`, subtract 1 at the end. The
`.max(0.0)` on the result guards the floating-point case where the subtraction of a
value marginally below 1 produces a small negative, which would be a nonsensical
branching ratio.

The shift is needed because power iteration does not converge on a **periodic**
(cyclic) matrix, where several eigenvalues share the maximum modulus. For

```text
A = [[0, 2], [0.9, 0]]        eigenvalues +-sqrt(1.8) = +-1.3416...
```

the iterate alternates between two directions and the bounds oscillate between `0.9` and
`2` forever. Adding `I` moves the eigenvalues to `1 +- 1.3416`, whose moduli are
`2.3416` and `0.3416` — no longer tied — so the iteration converges.

Measured: `1.341640786500` against a true `1.341640786500`, error `3.3e-15`, converged
in **17** iterations.

## 5. Why there is no "the upper bound stopped moving" exit

The only exit is bracket closure, `upper - lower <= 1e-14 * max(upper, 1)`, plus the
10 000-iteration cap as a backstop. An exit on `upper` having stopped decreasing looks
tempting — it is the quantity being returned — and is wrong.

`upper` is not monotone in a useful sense from step to step: it can plateau for several
iterations and then resume falling. On the nilpotent matrix

```text
N = [[0,1,0],[0,0,1],[0,0,0]]        rho(N) = 0
```

the shifted iteration produces, measured:

| `k` | `lower` | `upper` | `upper - 1` |
| --- | --- | --- | --- |
| 0 | 1.000000 | 2.000000 | **1.000000** |
| 1 | 1.000000 | 2.000000 | **1.000000** |
| 2 | 1.000000 | 1.750000 | 0.750000 |
| 3 | 1.000000 | 1.571429 | 0.571429 |
| 4 | 1.000000 | 1.454545 | 0.454545 |

`upper` sits at exactly `2.0` for two consecutive iterations before it begins to descend.
An early exit there returns `1.0` for a matrix whose spectral radius is `0` — a
trivially stationary process reported as sitting exactly on the explosive boundary. That
was a real bug in this routine, found by widening the test set to defective matrices
(CLAUDE.md §3), and it is why the exit condition is bracket closure alone.

A nilpotent `alpha` is a feed-forward cascade: component 1 excites 2, 2 excites 3, and
nothing feeds back. It is a natural model, not a pathology.

## 6. Accuracy, and why `3e-4` is enough

For a **diagonalizable** matrix the iterate converges geometrically at the rate given by
the ratio of the two largest eigenvalue moduli of `A + I`, and the routine reaches the
`1e-14` bracket-closure test long before the cap. Every hand-pinned case in
`spectral_radius.rs` is exact to `1e-9` or better.

For a **defective** matrix — a repeated eigenvalue with too few eigenvectors — the
convergence is sublinear. Writing the relevant Jordan block as `lambda*I + N` with
`N^2 = 0`, the shifted powers are

```text
(J + I)^k = (lambda+1)^k * I + k * (lambda+1)^(k-1) * N
```

so the `N` term is smaller than the diagonal term only by a factor `~ k`, and the ratio
`upper(x_k)` approaches `rho` like `C/k` rather than like a geometric rate. The error at
the cap is therefore `~ C / 10000`.

Measured on `[[0.4, 7], [0, 0.4]]`, taking the iteration cap as the variable:

| cap | `upper - rho` | `(upper - rho) * cap` |
| --- | --- | --- |
| 10 | 1.521739e-01 | 1.5217 |
| 100 | 1.411290e-02 | 1.4113 |
| 1 000 | 1.401121e-03 | 1.4011 |
| 10 000 | 1.400112e-04 | 1.4001 |

The product is constant to four figures, which is the `Theta(1/k)` law above rather than
an assertion that it holds. Across the defective cases the tests pin, the error at the
shipped cap is:

| matrix | `rho` | returned | error |
| --- | --- | --- | --- |
| `[[0.4, 7], [0, 0.4]]` | 0.4 | 0.400140011201 | 1.400e-04 |
| `[[0.4, 0.01], [0, 0.4]]` | 0.4 | 0.400138080679 | 1.381e-04 |
| `[[0, 3], [0, 0]]` | 0.0 | 0.000100006667 | 1.000e-04 |
| `N` (3×3 nilpotent) | 0.0 | 0.000200019998 | 2.000e-04 |

The largest measured error is `2.0e-04`, so "within about `3e-4`" is a statement with
headroom rather than a fitted bound. `DEFECTIVE_TOLERANCE` in `spectral_radius.rs` is
`1e-3`, looser again.

**Why that is enough.** The value decides `rho < 1` and is reported as a diagnostic. An
error of `2e-4` misleads only for a process whose true spectral radius lies within
`2e-4` of `1` — where the fit is on the stationarity boundary and the estimate's own
sampling error is orders of magnitude larger. The error is also **one-sided**: `upper` is
a rigorous upper bound at every step (§2), so the routine can report a stationary process
as marginally non-stationary but never the reverse. For a diagnostic guarding an
explosive process, that is the safe direction.

It would matter to a caller reading the number as a precise quantity, which is why the
doc comment states it rather than leaving it to be discovered.

## 7. What the tests pin

`hawk/tests/spectral_radius.rs`:

- diagonalizable cases with hand-computed radii, to `1e-9`
- the reducible/diagonal case of §3.1, which fails if the midpoint is returned
- the periodic case of §4, which does not terminate without the shift
- the defective cases of §6 at `1e-3`, including the nilpotent matrix that exposed the
  early-exit bug of §5

The last group was added after the first version of the file, which is how both bugs in
this routine were found — the §3 rule about widening the case set to the regime where
the method is weakest.
