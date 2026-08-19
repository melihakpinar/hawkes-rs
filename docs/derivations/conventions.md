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
