# Conventions, pinned to sources

Status: **findings recorded, awaiting owner approval.** Per CLAUDE.md §4 no
implementation may depend on this document until it is approved. M0 implements no
formula, so nothing depends on it yet.

Every statement below is traced to `tick`'s source or to an experiment run against
the pinned oracle image (`benchmarks/docker/`). Where a statement is empirical, the
experiment is described precisely enough to re-run. Where nothing settles a point, it
is not stated here — it is filed in `docs/open-questions.md`.

Source paths are relative to `site-packages/tick` in the pinned image
(`tick==0.8.0.2`, see `benchmarks/docker/README.md`).

---

## C1. Kernel normalization

```
phi_ij(t) = alpha_ij * beta_ij * exp(-beta_ij * t) * 1_{t > 0}
```

Source: `hawkes/model/model_hawkes_expkern_loglik.py:41-43`, the class docstring of
`ModelHawkesExpKernLogLik`:

> `\phi_{ij}(t) = \alpha^{ij} \beta^{ij} \exp (- \beta^{ij} t) 1_{t > 0}`

This is the `alpha * beta * exp(-beta t)` branch of the two conventions named in
CLAUDE.md §1.3, **not** `alpha * exp(-beta t)`. Consequently `alpha_ij` is
dimensionless and equals the integral of the kernel:

```
int_0^inf phi_ij(t) dt = alpha_ij
```

`tick` calls the matrix `(alpha_ij)` the **adjacency** matrix.

### Evidence (M1 Part A)

The docstring above is **not sufficient on its own**. C7 establishes that the same
docstring is wrong about what `loss` returns, so it cannot be trusted as the sole
source for anything else. Two independent experiments settle the kernel form, neither
of which uses the likelihood. Both are in `benchmarks/docker/convention_experiments.py`
(experiments E1 and E2).

**E1 — evaluate the kernel directly.** `HawkesKernelExp(intensity=0.35, decay=1.3)`
exposes `get_value`, `get_norm` and `get_primitive_value`, none of which involve a
likelihood:

| `t` | `get_value(t)` | `alpha*beta*exp(-beta t)` | `alpha*exp(-beta t)` |
| --- | --- | --- | --- |
| 0.0 | 0.45499999999999996 | **0.45499999999999996** | 0.35 |
| 0.5 | 0.23753082842626227 | **0.23753082842626227** | 0.1827160218663556 |
| 1.0 | 0.12400196583047572 | **0.12400196583047572** | 0.0953861275619044 |

`get_value(0)` alone is decisive: it returns `alpha*beta`, not `alpha`.

`get_norm()` returns `0.35 == alpha`, not `alpha/beta == 0.269...`, confirming C2
independently of the simulator.

`get_primitive_value(t)`, the kernel integral used by the compensator, matches
`alpha*(1 - exp(-beta t))` exactly at `t` in {0.5, 1.0, 3.0}.

**E2 — via `tick`'s gradient.** The gradient is the derivative of whatever `loss`
computes, so any *parameter-independent* offset differentiates away and the open
question in OQ-8 cannot contaminate the conclusion. E1c confirms that `tick`'s
analytic gradient agrees with central differences of `tick`'s own loss, which is what
licenses this route.

For `events = [1.0, 2.0, 2.5]`, `T = 3`, `mu = 0.7`, `alpha = 0.35`, `beta = 1.3`,
`tick` reports `d(loss*n)/dmu = -0.6399003563316408` and
`d(loss*n)/dalpha = 0.8395209932253471`. Predicting those from

```
d(loss*n)/dmu    = T - sum_k 1/lambda(t_k)
d(loss*n)/dalpha = sum_i dPhi(T-t_i)/dalpha - sum_k (1/lambda(t_k)) dlambda(t_k)/dalpha
```

gives `(-0.639900356331641, 0.8395209932253471)` under `alpha*beta*exp` and
`(-0.7581947432670377, 0.5843061858675322)` under `alpha*exp`. The first matches to
1e-9; the second is wrong in the third significant figure.

## C2. Branching ratio

Because the kernel integrates to `alpha_ij` (C1), the branching matrix *is* the
adjacency matrix, and stationarity requires

```
spectral_radius(alpha) < 1
```

Confirmed empirically: `SimuHawkesExpKernels(adjacency=[[0.2]], decays=[[1.5]], ...)`
reports `spectral_radius() == 0.2`, independent of the decay. A decay-dependent value
(e.g. `alpha/beta = 0.1333`) would indicate the other normalization. Reproduce with
`benchmarks/docker/smoke_test.py`, which prints the spectral radius.

## C3. Sum bounds in the intensity

Strict inequality: only events **strictly before** `t` contribute.

```
lambda_i(t) = mu_i + sum_{j=1}^{D} sum_{t_k^j < t} phi_ij(t - t_k^j)
```

Source: `hawkes/model/model_hawkes_expkern_loglik.py:31-33`, which writes
`\sum_{t_k^j < t}`. Same formula in
`hawkes/inference/base/learner_hawkes_param.py:29`.

This is the choice that keeps the intensity predictable (left-continuous), so the
intensity at the very first event is `mu_i`.

### Evidence (M1 Part A)

As with C1, the docstring is not sufficient on its own. Experiment E2b in
`benchmarks/docker/convention_experiments.py` settles it via the gradient.

First, a clarification that matters. **With distinct timestamps the strict and
inclusive readings are indistinguishable**: `t_i < t_k` and `t_i <= t_k excluding
i == k` select exactly the same set. The choice with observable content is whether
the jump *at* `t_k` contributes `phi(0) = alpha*beta` to `lambda(t_k)` — that is,
whether the intensity is predictable or not.

On the same data as E2, predicting `tick`'s gradient under each reading:

| reading | predicted `d/dmu` | predicted `d/dalpha` | |
| --- | --- | --- | --- |
| strict, predictable | -0.639900356331641 | 0.8395209932253471 | **matches tick** |
| self-inclusive | 0.6661227685190942 | -1.7725252564761234 | wrong sign |

The self-inclusive reading is not marginally wrong; it flips the sign of both
partials. The intensity is predictable and `lambda_i(t_1) = mu_i`.

E3b gives a second, independent confirmation using tied timestamps — see C8.

## C4. Compensator on the tail

The compensator integral runs to the observation horizon `T`, not to the last event.

Empirically pinned. With `adjacency == 0` the intensity is constant at `mu_i`, so the
compensator term is linear in `mu_i` with slope equal to the integration length. The
gradient `d loss / d mu_i` reported by `tick` was solved for that slope at
`T in {0.9, 1.0, 2.0, 5.0}` on a fixed 5-event, 2-node realization whose last event
is at `t = 0.9`. The recovered slope equalled `T` exactly in every case, including
`T = 5.0` where the last event is at `0.9`. Had the integral stopped at the last
event, the slope would have been pinned at `0.9` throughout.

## C5. Observation window

The caller supplies `[0, T]`. When `end_times=None`, `tick` infers
`T = max(events)`:

```python
# hawkes/model/base/model_hawkes.py:88-91
end_times = self._end_times
if end_times is None:
    non_empty_events = [[r for r in e if len(r) > 0] for e in events]
    end_times = np.array([max(map(max, e)) for e in non_empty_events])
```

This is the silent-bias case CLAUDE.md §1.3 warns about: discarding the trailing dead
time inflates the estimated baseline. Every fixture in `tests/fixtures/` therefore
passes `end_times` explicitly.

Note that `HawkesExpKern.fit()` accepts **no** `end_times` argument at all
(`hawkes/inference/base/learner_hawkes_param.py`, `fit(self, events, start=None)`),
so `tick`'s high-level learner always takes the inferred window. This is a real
difference in behaviour between `tick`'s learner and its model, and it will show up
when benchmarking baseline recovery.

## C6. Multivariate index order

`alpha[i][j]` means "**j excites i**".

Source: the intensity in `model_hawkes_expkern_loglik.py:31-33` sums `phi_ij` over
`j` for a fixed output component `i`; `i` is the excited node and `j` the exciting
node.

The flat coefficient vector `tick` optimizes is

```
coeffs = [ mu_0 .. mu_{D-1},  alpha_00, alpha_01, .., alpha_{D-1,D-1} ]
```

that is, the baseline block followed by the adjacency matrix raveled in **C (row-major)
order**:

- `hawkes/inference/base/learner_hawkes_param.py:227` — `return self.coeffs[:self.n_nodes]`
- `hawkes/inference/hawkes_expkern_fixeddecay.py:197-198` —
  `return self.coeffs[self.n_nodes:].reshape((self.n_nodes, self.n_nodes))`

That the reshaped matrix carries the same orientation as the simulator's `adjacency`
argument is confirmed by `hawkes/inference/hawkes_expkern_fixeddecay.py:200-202`,
which feeds the fitted `self.adjacency` straight back into
`SimuHawkesExpKernels(adjacency=...)`.

## C7. What `tick`'s `loss` actually returns

**Not** the negative log-likelihood, and not the quantity in its own docstring.

`ModelHawkesExpKernLogLik`'s docstring (`model_hawkes_expkern_loglik.py:17-21`)
advertises

```
sum_i ( int_0^T lambda_i(t) dt - int_0^T log lambda_i(t) dN_i(t) )
```

The value returned by `model.loss(coeffs)` differs from that. Measured on the
`adjacency == 0` case, where the model collapses to a homogeneous Poisson process
with the closed form `sum_i ( mu_i*T - n_i*log mu_i )`:

```
loss * n_jumps - sum_i ( mu_i*T - n_i*log mu_i )  ==  -D*T
```

exactly, across `D in {1, 2, 3}`, `T in {1, 2}` and several `mu` vectors. The
identity that reproduces this is the negative log-likelihood **ratio against a
unit-rate Poisson process**, normalized by the total jump count:

```
loss = (1/n_jumps) * sum_i [ int_0^T ( lambda_i(t) - 1 ) dt - sum_k log lambda_i(t_k) ]
```

since `int_0^T sum_i 1 dt = D*T`.

Related: `LearnerHawkesParametric.score()` is exactly `-model.loss(coeffs)`
(`hawkes/inference/base/learner_hawkes_param.py:332`), so it inherits both the
normalization and the offset.

**This is confirmed only for `adjacency == 0`.** The `-D*T` term is predicted to be
parameter-independent under the ratio interpretation, but that has not been tested
with excitation present, because doing so requires a Hawkes likelihood that does not
exist yet. Filed as `docs/open-questions.md` OQ-8; it is decidable by a differential
test as soon as M1 lands a likelihood, and it must be decided before any absolute
log-likelihood comparison against `tick` is trusted.

---

## C8. Event ordering, exact ties, and `hawk`'s input contract

Settled by experiment E3 in `benchmarks/docker/convention_experiments.py`. This
closes OQ-7.

### What `tick` does

**Order matters, and `tick` does not check it.** The same three timestamps in three
orders, with identical parameters, `T` and `n_jumps` — so any loss offset is
identical on all three and cancels in the comparison:

| input array | `loss` |
| --- | --- |
| `[1.0, 2.0, 2.5]` (sorted) | 0.13129881231404172 |
| `[2.5, 2.0, 1.0]` (reversed) | 0.30530897667097345 |
| `[2.0, 1.0, 2.5]` (shuffled) | 0.18566300584064774 |

Three different answers, no error, no warning. `tick` requires per-component
ascending order and silently returns a wrong number when it does not get it. This is
the single most dangerous behaviour found in `tick` so far: unlike OQ-9, which fails
loudly at construction, this one produces a plausible number from malformed input.

**Ties are accepted, and tied events do not excite each other.** For
`events = [1.0, 2.0, 2.0]`, `tick` reports `d(loss*n)/dmu = -0.855750104030397`.
Predicting that value:

| rule for the tied pair | predicted `d/dmu` | |
| --- | --- | --- |
| neither tied event excites the other | -0.8557501040303972 | **matches tick** |
| the earlier-indexed tied event excites the later | -0.42402039335057484 | |

So a tie is resolved by the strict inequality `t_i < t_k` on *times*, not by array
position. This is exactly C3's predictable convention applied at zero lag, and it is
an independent confirmation of C3 on data where the two readings genuinely differ.

Cross-component ties (`[[1.0], [1.0]]`) are accepted. An empty component in a
multivariate process is accepted. Timestamps at exactly `0.0` and at exactly `T` are
accepted; a timestamp beyond `T` is rejected with

```
RuntimeError: Provided end_time (3) is smaller than last time of component 0 (4)
```

### `hawk`'s input contract

`tick`'s behaviour constrains the *mathematics*; it does not oblige `hawk` to copy
`tick`'s error handling. Where the two differ below, the reason is given.

1. **Timestamps must be sorted ascending within each component.** `hawk` **rejects**
   unsorted input with an error rather than silently computing the wrong answer as
   `tick` does. CLAUDE.md §5 requires invalid input to be an error value, and the
   experiment above is the argument for treating unsorted input as invalid rather
   than as something to quietly sort: a caller who supplies unsorted data has
   probably misunderstood their own data, and sorting it for them hides that.

2. **Cross-component order is not defined and not required.** Components are
   separate sequences; there is no global ordering constraint. Nothing in the
   intensity depends on how events of different components interleave in storage,
   only on their times.

3. **Exact ties are accepted**, within a component and across components, and are
   evaluated under C3: an event never contributes to the intensity at its own time,
   nor to that of any simultaneous event. `hawk` matches `tick` here because the
   convention is forced by C3 rather than chosen, and because tied timestamps are
   common in real data as an artifact of finite clock resolution. A tie means the
   recursion must group by *distinct* time — see
   `docs/derivations/univariate_loglikelihood.md` §5, where the naive
   event-to-event recursion is shown to be wrong in exactly this case.

4. **Every timestamp must lie in `[0, T]`**, endpoints included. `T` is supplied by
   the caller and never inferred (C5).

5. **A component may be empty.** With no events at all the log term is empty and the
   likelihood is that of a homogeneous Poisson process.
