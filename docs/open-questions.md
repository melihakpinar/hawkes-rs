# Open questions

Every unresolved convention, index range or numerical choice lives here, per
CLAUDE.md §1.2. An entry records the question, what was searched and read, the
candidate resolutions, and what observably differs between them.

Statuses:

- **OPEN** — not settled; work touching it must not proceed on a guess.
- **BLOCKED** — nothing available can decide it; the work item is stopped.
- **RESOLVED** — settled by a cited source or a decisive experiment. The resolution
  is recorded in `docs/derivations/conventions.md` and the entry is kept for the
  audit trail.

`tick` source paths are relative to `site-packages/tick` in the pinned oracle image
(`tick==0.8.0.2`; see `benchmarks/docker/README.md`).

The first seven entries are the convention hazards enumerated in CLAUDE.md §1.3.
They were opened at the start of M0 and each was then either settled from `tick`'s
source or left open.

---

## OQ-1 — Kernel normalization · RESOLVED (M0)

**Question.** Is the exponential kernel `alpha*exp(-beta t)` or
`alpha*beta*exp(-beta t)`?

**Searched.** `tick` class docstrings for `ModelHawkesExpKernLogLik` and
`SimuHawkesExpKernels`; `HawkesKernelExp`.

**Resolution.** `alpha*beta*exp(-beta t)`, from
`hawkes/model/model_hawkes_expkern_loglik.py:41-43`. Recorded as conventions.md C1.

---

## OQ-2 — Branching ratio · RESOLVED (M0)

**Question.** Under the chosen normalization, is the branching ratio `alpha/beta` or
`alpha`?

**Observable difference.** The two differ by a factor of `beta`, so they disagree
about stationarity for any `beta != 1`.

**Resolution.** `alpha`. The kernel of OQ-1 integrates to `alpha`, and `tick`'s
`spectral_radius()` returns `0.2` for `adjacency=[[0.2]], decays=[[1.5]]` — a
decay-independent value, which `alpha/beta = 0.1333` would not be. Recorded as
conventions.md C2.

---

## OQ-3 — Sum bounds in the intensity · RESOLVED (M0)

**Question.** Strict `t_j < t` or inclusive `t_j <= t`? What is the intensity at the
very first event?

**Observable difference.** Inclusive bounds make the intensity non-predictable and
add `phi_ii(0)` to the intensity at each of that node's own events, shifting the
log-likelihood.

**Resolution.** Strict: `hawkes/model/model_hawkes_expkern_loglik.py:31-33` writes
`\sum_{t_k^j < t}`. The intensity at the first event is therefore `mu_i`. Recorded as
conventions.md C3.

---

## OQ-4 — Compensator on the tail · RESOLVED (M0)

**Question.** Does the compensator integral run to `T` or to the last event time?

**Observable difference.** They differ by `sum_i int_{t_last}^{T} lambda_i(t) dt`,
which is not a constant — it grows with the dead time at the end of the window.

**Resolution.** To `T`. Pinned empirically by solving `tick`'s gradient for the
coefficient of `mu_i` at `adjacency == 0`; it equals `T` exactly for
`T in {0.9, 1.0, 2.0, 5.0}` on a realization whose last event is at `0.9`. Recorded
as conventions.md C4.

---

## OQ-5 — Observation window · RESOLVED (M0)

**Question.** Is `[0, T]` given by the caller or inferred as `[0, max(events)]`?

**Resolution.** `tick`'s model takes it from the caller and falls back to
`max(events)` when `end_times is None`
(`hawkes/model/base/model_hawkes.py:88-91`). All fixtures pass it explicitly.

Note for M1: `tick`'s *learner* (`HawkesExpKern.fit`) exposes no `end_times`
parameter at all, so it always uses the inferred window and its baseline estimates
are biased upward whenever the window has trailing dead time. Any benchmark
comparing baseline recovery against `HawkesExpKern` must account for this rather
than treat it as a `hawk` defect. Recorded as conventions.md C5.

---

## OQ-6 — Multivariate index order · RESOLVED (M0)

**Question.** Does `alpha[i][j]` mean "j excites i" or "i excites j"?

**Observable difference.** A transposed matrix yields a fitted result that looks
plausible and is wrong; it is only detectable on an asymmetric process.

**Resolution.** "j excites i". The intensity for output node `i` sums `phi_ij` over
`j` (`hawkes/model/model_hawkes_expkern_loglik.py:31-33`), and the flat coefficient
vector is `[baseline (D), adjacency raveled in C order (D*D)]`
(`hawkes/inference/base/learner_hawkes_param.py:227`,
`hawkes/inference/hawkes_expkern_fixeddecay.py:197-198`). Recorded as
conventions.md C6.

The fixtures `bivariate_asymmetric` and `trivariate_asymmetric` exist specifically to
make a transposition detectable.

---

## OQ-7 — Event ordering across dimensions and exact ties · OPEN

**Question.** Must timestamps be globally sorted, or only sorted within each
component? What must happen on exact ties, either between two components or within
one?

**Searched.** `hawkes/model/base/model_hawkes.py` (`_set_data`) performs no sorting
and no validation. The ordering requirement lives in `tick`'s C++ (`ModelHawkesExpKernLogLik::set_data`),
which is not shipped in the wheel — only the compiled extension and the SWIG shim
are present — so it cannot be cited by line from the pinned image.

**Candidates.**
(a) Per-component sorting is required; cross-component order is irrelevant.
(b) Global sorting is required.
(c) Ties are permitted and each tied event contributes; the recursion treats
    `exp(-beta * 0) = 1`.
(d) Ties are rejected.

**Observable difference.** Under strict bounds (OQ-3), two events at exactly the same
time on the same node do not excite each other, so (c) and (d) give different event
counts but the same intensity path. Feeding deliberately unsorted input and comparing
`loss` against the sorted-input `loss` distinguishes (a) from (b).

**Decidable by experiment**, and the experiment does not need a `hawk` likelihood:
feed `tick` the same events sorted and unsorted, and with and without a duplicated
timestamp, and compare `loss`. Not run in M0 because no `hawk` code consumes event
ordering yet. **Must be closed before the M1 likelihood lands**, since the input
contract for `hawk`'s public API depends on it.

---

## OQ-8 — Is `tick`'s loss offset parameter-independent? · OPEN

**Question.** `tick`'s `ModelHawkesExpKernLogLik.loss` is not its documented formula.
Measured at `adjacency == 0`:

```
loss * n_jumps - sum_i ( mu_i*T - n_i*log mu_i )  ==  -D*T
```

exactly, for `D in {1,2,3}`, `T in {1,2}` and several `mu`. The identity that
explains it is the negative log-likelihood *ratio against a unit-rate Poisson
process*, normalized by `n_jumps`:

```
loss = (1/n_jumps) * sum_i [ int_0^T ( lambda_i(t) - 1 ) dt - sum_k log lambda_i(t_k) ]
```

Does the `-D*T` offset remain exactly `-D*T` when `adjacency != 0`?

**Candidates.**
(a) Yes — the offset is `int_0^T sum_i 1 dt`, structurally independent of parameters.
(b) No — the offset also absorbs something excitation-dependent, and the ratio
    interpretation is a coincidence of the Poisson case.

**Observable difference.** Under (a) a correct `hawk` negative log-likelihood
satisfies `hawk_nll == tick_loss * n_jumps + D*T` for every fixture. Under (b) that
identity fails on the fixtures with `adjacency != 0` while still holding at
`adjacency == 0`.

**Why not resolved in M0.** Deciding it requires evaluating the Hawkes
log-likelihood with excitation present, independently of `tick`. No such
implementation exists yet, and writing one is out of M0's scope (M0 ships zero
algorithm code).

**How it gets closed.** The differential harness
(`hawk/tests/differential_tick.rs`) already replays every fixture at four parameter
points each, including three with `adjacency != 0`. The moment M1 lands a
log-likelihood, that harness decides this question. Until it is closed, **no
absolute log-likelihood comparison against `tick` may be trusted**; only differences
of the loss at two parameter points on the same data are safe, since the offset
cancels.

---

## OQ-9 — `tick` has an undeclared runtime dependency on numpydoc · RESOLVED (M0)

**Question.** Why does every `tick` model class fail to construct on a clean install
with `AttributeError: object has no settable attribute 'dtype'`?

**Searched.** `tick/base/base.py`. `BaseMeta.extract_attrinfos` registers an
attribute as settable only if it appears as a documented attribute, a property, or an
`__init__` parameter. `find_documented_attributes` (`base.py:256-264`) begins:

```python
if '__doc__' not in attrs or docscrape is None:
    return []
```

`docscrape` is `numpydoc.docscrape`, imported defensively at module top. `tick`
0.8.0.2 declares `numpydoc` only under the `docs` extra, not as a runtime
dependency. Without it every documented attribute — including `Model.dtype` — is
unregistered and construction raises.

**Resolution.** `numpydoc` is a hard runtime dependency and is pinned in
`benchmarks/docker/requirements.txt`. `pip install --no-deps numpydoc==1.9.0` is
sufficient; its own Sphinx dependency chain is not needed for `docscrape`.

This is a bug in `tick`'s packaging, not in this project. It is worth reporting
upstream.

---

## OQ-10 — The premise that `tick` is unusable on modern Python is falsified · OPEN

**Question.** CLAUDE.md's preamble states this project exists because `tick` "is
unmaintained and its estimators break on Python 3.13+". M0 found that `tick`
0.8.0.2 (published with wheels through CPython 3.14, `requires_python >= 3.11`)
imports, simulates and fits successfully on CPython 3.13.5, once OQ-9's missing
dependency is installed. The oracle image does exactly this and its smoke test
passes.

**What is and is not falsified.**

- Falsified: "breaks on Python 3.13+". It does not.
- Falsified: "unmaintained". 0.8.0.0 through 0.8.0.2 postdate the 0.7.0.1 release
  the premise appears to refer to, and they add 3.11-3.14 wheels.
- Not assessed: "no Rust crate does estimation at all". Not investigated in M0.
- Standing regardless: `tick` ships real defects (OQ-9; several `__repr__`
  implementations raise on unfitted objects; `HawkesExpKern` cannot express an
  observation window, OQ-5; `gofit="likelihood"` is unusable through the documented
  learner interface because it installs a prox that admits negative coefficients,
  and the C++ model then rejects them).

**Why this is filed rather than acted on.** It is a question about the project's
rationale and scope, which is the repository owner's call, not a mathematical
question this agent should resolve. M0's deliverables are unaffected either way: the
verification machinery is what makes `tick` usable as an oracle, and a *maintained*
oracle makes the differential tests stronger, not weaker.

**Needs a decision from the owner** on whether v0.1.0's positioning should change.
