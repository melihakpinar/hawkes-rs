# Hand calculations for the input-contract edge cases

Expected values for `hawkes/tests/input_contract.rs`, worked from the **definitions**
rather than from any recursion, so the tests that use them are independent of the
code under test (CLAUDE.md §3). Each is a direct substitution into an equation that is
already approved:

- univariate negative log-likelihood, `univariate_loglikelihood.md` (3.3);
- univariate gradient, `univariate_gradient.md` (G.1), (G.2), (G.7);
- univariate compensator, `univariate_loglikelihood.md` (2.1) evaluated at `t_k`
  instead of `T`, with the sum over `t_i < t_k` (C3);
- multivariate negative log-likelihood, `multivariate_loglikelihood.md` (M3.2);
- multivariate compensator, `multivariate_loglikelihood.md` §2, `Lambda_i(t)`.

Decimal values were evaluated with CPython's `math` module, which is not `hawkes`.

## A. One event at exactly the horizon

`mu = 1.5`, `alpha = 0.4`, `beta = 2`, `T = 4`, `t = [4]`.

The event sits at `T`, so its compensator term is `1 - exp(-beta*(T - T)) = 0`, and
nothing precedes it, so `lambda(t_1) = mu` (C3). From (3.3):

```
nll = mu*T + alpha*0 - ln(mu) = 6 - ln(1.5) = 5.594534891891835
```

`alpha` and `beta` do not appear in that expression, so (G.2) and (G.7) are exactly
zero, and (G.1) is

```
d nll / d mu = T - 1/mu = 4 - 0.6666666666666666 = 3.3333333333333335
```

## B. One event at exactly zero

Same parameters, `t = [0]`.

```
nll = mu*T + alpha*(1 - e^{-beta*T}) - ln(mu)
    = 6 + 0.4*(1 - e^{-8}) - ln(1.5)
    = 6 + 0.4*0.9996645373720975 - 0.4054651081081644
    = 5.994400706840675
```

Gradient, from (G.1), (G.2) and (G.7) with `B_1 = 0`, `Bp_1 = 0`, `W_1 = T`:

```
d nll / d mu    = T - 1/mu                 = 3.3333333333333335
d nll / d alpha = 1 - e^{-beta*T}           = 0.9996645373720975
d nll / d beta  = alpha * T * e^{-beta*T}   = 0.4 * 4 * e^{-8} = 0.000536740204644019
```

## C. Two components, one event at the horizon, the other component empty

`mu = [0.8, 0.4]`, any `alpha`, any `beta`, `T = 5`, `events = [[5], []]`.

From (M3.2): the compensator is `sum_i mu[i]*T = 6`; the only event is at `T`, so
every `1 - exp(-beta*(T - t))` term is zero; its intensity is `mu[0]` because nothing
precedes it.

```
nll = 6 - ln(0.8) = 6.223143551314211
d nll / d mu[0] = T - 1/mu[0] = 5 - 1.25 = 3.75
d nll / d mu[1] = T           = 5
```

Every `d nll / d alpha[i][j]` and `d nll / d beta` is zero, for the reason in A.

## D. Univariate compensator at tied events

`mu = 0.7`, `alpha = 0.5`, `beta = 1.3`, `t = [1, 2, 2, 3]`, any `T >= 3`.

`Lambda(t) = mu*t + alpha * sum_{t_i < t} (1 - e^{-beta*(t - t_i)})`, strictly over
earlier events, so the two events at `2` do not contribute to each other's value.

```
Lambda(1) = 0.7
Lambda(2) = 1.4 + 0.5*(1 - e^{-1.3})                       = 1.7637341034829936
Lambda(2) = the same                                        = 1.7637341034829936
Lambda(3) = 2.1 + 0.5*((1 - e^{-2.6}) + 2*(1 - e^{-1.3}))  = 3.29033141785882
```

## E. Multivariate compensator with a cross-component tie

`mu = [0.2, 0.5]`, `alpha = [[0.1, 0.6], [0.05, 0.15]]`, `beta = 1.2`, `T = 6`,
`events = [[1, 2.5], [2.5, 4]]` — the `§4.1` counterexample's data.

`Lambda_i(t) = mu[i]*t + sum_j alpha[i][j] * sum_{t^j_l < t} (1 - e^{-beta*(t - t^j_l)})`.
At `t = 2.5` the other component's event at `2.5` is not earlier and contributes
nothing.

```
Lambda_0(1)   = 0.2
Lambda_0(2.5) = 0.5 + 0.1*(1 - e^{-1.8})                          = 0.5834701111778413
Lambda_1(2.5) = 1.25 + 0.05*(1 - e^{-1.8})                        = 1.2917350555889207
Lambda_1(4)   = 2.0 + 0.05*((1 - e^{-3.6}) + (1 - e^{-1.8})) + 0.15*(1 - e^{-1.8})
                                                                   = 2.215574036233318
```
