# Which parameter space the fit optimizes in

Status: **decision recorded before implementation**, per M2 Part B constraint (a).

CLAUDE.md §6 says positivity is handled by optimizing in log-parameter space rather
than by constrained optimization. In one dimension that settled the question. In `d`
dimensions it does not, because of `alpha`.

## The problem

`mu[i] > 0` and `beta > 0` are strict: a baseline of zero means a component with no
exogenous arrivals, and a decay of zero is not an exponential kernel. `ln` is defined
on both.

`alpha[i][j] >= 0` is **not** strict. A zero entry means "component `j` does not excite
component `i`", which is an ordinary thing for a `d`-component process to be — a
block-diagonal `alpha` is two independent processes, a zero row is a component nobody
influences. `ln 0` is not defined.

This did not arise in M1: a univariate `alpha` of zero is a homogeneous Poisson process,
a degenerate case rather than a structural one.

## The decision

**Optimize `(ln mu, ln alpha, ln beta)` — log space for all three, `alpha` included.**

## Why

### The alternative needs machinery, and `tick` shows what happens without it

Natural-space optimization with `alpha >= 0` is a bound-constrained problem. `argmin`'s
L-BFGS is unconstrained, so it would need a projection or an active-set method written
here.

`tick` attempted the unconstrained route and it does not work: `penalty="none"` installs
a prox that admits negative coefficients, and its C++ model then rejects them with
*"The sum of the influence on someone cannot be negative"* — the failure recorded in
`docs/positioning-probe.md` §5.2, which makes `tick`'s documented likelihood-fitting
interface unusable. Reproducing that shape of bug is not attractive.

### The excluded region is not where the optimum is

The concern with log space is that `alpha[i][j] = 0` becomes unreachable: the fit can
approach zero but never return it. That would matter if the maximum-likelihood estimate
were ever exactly zero. It is not.

The unpenalized log-likelihood is smooth in `alpha` and its stationary point is interior
almost surely. Even when the **true** `alpha[i][j]` is zero, the estimate is not: it is
a positive random variable of order `T^{-1/2}`, and the optimizer reaches
`ln alpha ~= -0.5 * ln T`, which is a modest and perfectly well-behaved number — about
`-4` at `T = 3000`. Log space does not push the optimizer anywhere it does not already
want to go.

Exact zeros in a *fitted* matrix are the business of L1 or nuclear regularization, which
CLAUDE.md §8 puts out of scope for v0.1.0. When that arrives it will need a
proximal-gradient method and this decision must be revisited with it, because a prox
operator for L1 acts in the natural parametrization.

### What is given up, stated plainly

- **A fit never returns an exact zero.** Every entry of a fitted `alpha` is strictly
  positive. A caller reading a fitted matrix as a sparsity pattern must threshold it,
  and the threshold is theirs to choose; `hawk` does not choose one.
- **A true zero is recovered as a small positive number**, not as zero. The round-trip
  test accounts for this by comparing against the estimate's own standard error, which
  is the right scale for that comparison, rather than against a fixed tolerance that
  would have to be absolute near zero and relative away from it.
- **`ln alpha` is unbounded below**, so a component pair with no interaction at all
  drives its coordinate steadily negative. The iteration cap stops it. The reported
  `converged` flag is what tells a caller this happened; it is a gradient measurement,
  not a claim by the optimizer (M1's lesson, `docs/positioning-probe.md` part 3).

## Consequences for the input side

`Parameters::new` continues to **accept** `alpha[i][j] = 0`. The decision above is about
the optimizer's coordinates, not about the model's domain: evaluating the likelihood,
the gradient, the compensator and the simulator at an exact zero is legitimate and
tested (`multivariate_loglikelihood.rs::agrees_with_brute_force_on_structural_zeros`,
`multivariate_gradient.rs::matches_with_structural_zeros`).

Only `fit` is constrained, and only in where it can land.

## Verification

| Check | Covers |
| --- | --- |
| Round-trip recovery elementwise on `alpha`, `d` in 2..=5 | that the log-space coordinates reach the right place |
| A true `alpha[i][j] = 0` entry, recovered within its standard error | the boundary case this decision is about |
| Multi-start invariance | that the coordinate choice does not create local optima the fit gets stuck in |
| Sabotage: treat the log coordinate as natural for `alpha` | that the conversion at the boundary is actually exercised |
