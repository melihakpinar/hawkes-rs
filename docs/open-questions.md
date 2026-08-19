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

## OQ-1 — Kernel normalization · RESOLVED (M0, re-grounded M1 Part A)

**Question.** Is the exponential kernel `alpha*exp(-beta t)` or
`alpha*beta*exp(-beta t)`?

**Searched.** `tick` class docstrings for `ModelHawkesExpKernLogLik` and
`SimuHawkesExpKernels`; `HawkesKernelExp`.

**Resolution.** `alpha*beta*exp(-beta t)`, from
`hawkes/model/model_hawkes_expkern_loglik.py:41-43`. Recorded as conventions.md C1.

**Re-grounded (M1 Part A).** The M0 resolution rested on a class docstring, and OQ-8
showed the *same docstring* is wrong about what `loss` returns. A docstring that is
wrong once cannot be the sole source for anything. Two experiments now settle it
without touching the likelihood (`benchmarks/docker/convention_experiments.py`):

- **E1** evaluates the kernel directly. `HawkesKernelExp(0.35, 1.3).get_value(0)`
  returns `0.455 == alpha*beta`, not `0.35 == alpha`; `get_value(t)` matches
  `alpha*beta*exp(-beta t)` at every `t` tested; `get_norm()` returns `alpha`.
- **E2** predicts `tick`'s gradient, which is immune to any parameter-independent
  loss offset, and matches only under `alpha*beta*exp`.

conventions.md C1 now cites both.

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

## OQ-3 — Sum bounds in the intensity · RESOLVED (M0, re-grounded M1 Part A)

**Question.** Strict `t_j < t` or inclusive `t_j <= t`? What is the intensity at the
very first event?

**Observable difference.** Inclusive bounds make the intensity non-predictable and
add `phi_ii(0)` to the intensity at each of that node's own events, shifting the
log-likelihood.

**Resolution.** Strict: `hawkes/model/model_hawkes_expkern_loglik.py:31-33` writes
`\sum_{t_k^j < t}`. The intensity at the first event is therefore `mu_i`. Recorded as
conventions.md C3.

**Re-grounded (M1 Part A).** Same objection as OQ-1: the docstring is not a
sufficient source. Experiment E2b also sharpened the question. With **distinct**
timestamps, `t_i < t_k` and `t_i <= t_k excluding i == k` select the same set and are
therefore indistinguishable; the choice with observable content is whether the jump
*at* `t_k` adds `phi(0) = alpha*beta` to `lambda(t_k)`.

Predicting `tick`'s gradient under each reading: the predictable form matches to
1e-9, while the self-inclusive form gets `d/dmu = +0.666` against `tick`'s `-0.640`
— the wrong sign on both partials. Experiment E3b confirms it a second way, on tied
timestamps, where the two readings genuinely differ.

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

## OQ-7 — Event ordering across dimensions and exact ties · RESOLVED (M1 Part A)

**Question.** Must timestamps be globally sorted, or only sorted within each
component? What must happen on exact ties, either between two components or within
one?

**Searched (M0).** `hawkes/model/base/model_hawkes.py` (`_set_data`) performs no
sorting and no validation. The ordering requirement lives in `tick`'s C++, which the
wheel does not ship, so it cannot be cited by line.

**Resolved by experiment E3** in `benchmarks/docker/convention_experiments.py`,
exactly as this entry specified. Loss values are compared only across runs sharing
parameters, `D`, `T` and `n_jumps`, so any OQ-8 offset is identical on both sides and
cancels.

**(a) vs (b) — ordering.** Candidate (a): per-component ordering is what matters. The
same three timestamps in three orders give three different losses — 0.13129881,
0.30530898, 0.18566301 — with **no error and no warning**. Cross-component order is
not expressible in `tick`'s API (components are separate arrays) and nothing in the
intensity depends on it.

This is the most dangerous `tick` behaviour found so far. OQ-9 fails loudly at
construction; this one turns malformed input into a plausible wrong number.

**(c) vs (d) — ties.** Candidate (c): ties are accepted, and tied events do **not**
excite each other. On `[1.0, 2.0, 2.0]`, `tick`'s `d(loss*n)/dmu` is
`-0.855750104030397`, matching the "neither tied event excites the other" prediction
to 1e-9; the alternative predicts `-0.424020`. The tie is resolved by the strict
inequality on *times*, not by array position — C3 applied at zero lag.

Also observed: cross-component ties accepted, empty components accepted, timestamps
at exactly `0.0` and exactly `T` accepted, timestamps beyond `T` rejected with a
clear error.

**`hawk`'s input contract** is stated in `docs/derivations/conventions.md` C8. It
differs from `tick` in one deliberate place: `hawk` **rejects** unsorted input rather
than silently computing the wrong answer.

**Consequence for the implementation.** Admitting ties makes the textbook Ozaki
recursion `A_k = exp(-beta*(t_k - t_{k-1}))*(1 + A_{k-1})` **incorrect**: at a tie it
evaluates `exp(0) = 1` and counts the simultaneous event as exciting. The recursion
must group by distinct time. Derived and demonstrated in
`docs/derivations/univariate_loglikelihood.md` §4.2, where the textbook form is off
by about 9% on a four-event example.

---

## OQ-8 — Is `tick`'s loss offset parameter-independent? · OPEN (candidate (a) confirmed by independent evidence; formal closure in M1 Part B step 9)

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

**Preliminary evidence (M1 Part A) — candidate (a).** `benchmarks/docker/oq8_preliminary.py`
evaluates the multivariate negative log-likelihood directly from the definition, in
Python, using no `hawk` code, and tests

```
tick_loss * n_jumps + D*T  ==  nll
```

on **all six committed fixtures at all 24 parameter points, 18 of which have
`adjacency != 0`**. Every one matches, with absolute residuals from `0.0` to
`4.6e-11` against `nll` values in the hundreds to thousands — machine precision.

That is candidate (a): the offset is exactly `-D*T`, independent of the parameters,
with excitation present. Experiment E2 supports it from a second direction, since
`tick`'s gradient matches the derivative of that same `nll` at `alpha != 0`, which
means the difference between the two is constant in the parameters.

**Left OPEN deliberately.** M1's plan closes this at Part B step 9 with `hawk`'s own
implementation, after that implementation has been validated against non-`tick`
oracles. The Python reference above shares an author with the derivation and has not
been sabotage-tested; it is evidence, not the verdict. Nothing is expected to change,
but the entry stays open until the planned test runs.

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

---

## OQ-11 — Reference PDFs are absent, so equations cannot be cited by number · OPEN

**Question.** CLAUDE.md §2 requires papers to be cited by equation number
(`[Ozaki1979, eq. 7]`). `docs/references/` contains only a README: the PDFs are under
publisher copyright and are not committed, and I do not have them.

**What was done instead.** `docs/derivations/univariate_loglikelihood.md` and
`univariate_gradient.md` are written to be **self-contained**: every step is shown,
so both are checkable line by line without any paper. [Ozaki1979] is named for
provenance of the recursion technique, without an equation number, and §0 of the
log-likelihood derivation says so explicitly rather than burying it.

**What was not done.** No equation number was invented. Citing `[Ozaki1979, eq. 7]`
without having read equation 7 would be precisely the fabricated-source failure
CLAUDE.md §1 exists to prevent, and it would be undetectable by a later reader.

**Resolution path.** Obtain the PDFs listed in `docs/references/README.md`, place
them there, and fill in the equation numbers in both derivations. Purely an audit-trail
task: the mathematics is independently verified by
`docs/derivations/check_univariate_derivation.py` and, from Part B, by the Rust
oracles.

**Needs the repository owner** — obtaining copyrighted PDFs is not something this
agent can do.
