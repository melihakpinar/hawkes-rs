# Architecture review, September 2026

Review only. Nothing here changes code; each recommendation worth acting on has a
GitHub issue, linked in place, and the rest are recorded as decisions not to act.

Line references are to the tree at the merge of #50. Timings below marked *measured for
this review* were taken on the benchmark machine of `benchmarks/README.md` §2 (Apple M2,
`cargo build --release`, single-threaded, median of 7 after 1 warmup) on
`univariate::simulate` output with `mu = 0.5, alpha = 0.6, beta = 1.0`, seed 20260819,
the positioning probe's data. They are one run each and are cited as evidence of
magnitude, not as benchmark results; the committed numbers remain those in
`benchmarks/results/`.

## 1. The univariate and multivariate modules

### Observation

`hawkes/src/univariate.rs` (756 lines) and `hawkes/src/multivariate.rs` (1218 lines)
are two implementations of the same model. They share three inlined helpers —
`advance_excitation_state`, `intensity_at`, `compensator_contribution`
(`univariate.rs:163-201`) — and nothing else. Each has its own `Parameters`,
`Observation`, `Gradient`, `Fit`, `negative_log_likelihood`,
`negative_log_likelihood_and_gradient`, `compensator_at_events`, `simulate` and `fit`.
At `d = 1` the two agree **bitwise** on value and gradient
(`hawkes/tests/multivariate_equivalence.rs`, 600 randomised cases plus the fixture
corpus in `differential_tick.rs::univariate_path_agrees_on_the_univariate_fixtures`),
which is a deliberate property maintained by two accumulation rules in
`multivariate_loglikelihood.md` §5.1.

`lib.rs:6-8` states the reason for keeping both: "`univariate` is kept because it is
measurably cheaper and is the reference the multivariate path is checked against."

### Evidence

Measured for this review, same events through both paths, `d = 1`:

| events | value only, uni / multi | value + gradient, uni / multi | `fit`, uni / multi | objective evaluations, uni / multi |
| --- | --- | --- | --- | --- |
| 10 079 | 0.40 ms / 0.60 ms (1.52x) | 0.76 ms / 0.61 ms (0.81x) | 9.2 ms / 9.0 ms | 12 / 12 |
| 100 041 | 1.5 ms / 2.3 ms (1.49x) | 2.6 ms / 2.6 ms (1.01x) | 85 ms / 107 ms | 13 / 15 |
| 998 244 | 15.5 ms / 22.5 ms (1.45x) | 26.3 ms / 26.2 ms (1.00x) | 0.85 s / 3.44 s (4.03x) | 13 / 63 |

Three things follow.

- The value-only kernel, which is what every line-search trial evaluates
  (`docs/positioning-probe.md` part 2), is about 1.5x cheaper in `univariate`. The
  value-plus-gradient kernel is not cheaper at all: the `d = 1` multivariate loop
  costs the same as the univariate one. So "measurably cheaper" is true of half the
  hot path.
- **The two `fit`s are not the same optimizer.** At a million events the multivariate
  fit makes 63 objective evaluations against 13, and takes four times as long, on
  identical data and identical starting points. The only configuration differences
  are the L-BFGS memory (7 at `univariate.rs:704`, 10 at `multivariate.rs:1045`) and
  the iteration cap (500 at `:719`, 1000 at `:1052`); the line search, tolerances,
  scaling and starting point are the same. Which of those, or what else, produces a
  five-fold difference in line-search trials is not established, and it means the
  positioning probe's "36 passes against `tick`'s 23" describes the univariate fit
  only.
- The bit-identity test is what makes the duplication safe. Two independent
  implementations that must agree to the bit are a stronger check than one
  implementation and a wrapper, because the wrapper would agree with itself
  vacuously. Collapsing `univariate` into `multivariate` at `d = 1` would delete the
  oracle along with the duplication.

The fit drivers are duplicated in the sense CLAUDE.md §5 cares about: the `Problem`
struct and its two trait impls (`univariate.rs:615-681`, `multivariate.rs:974-1028`),
the line-search and solver assembly (`:697-708`, `:1040-1049`), the post-hoc
convergence measurement (`:734-741`, `:1069-1078`) and the `Fit` struct (seven fields,
`:529-564`, `:877-889`) all encode one set of decisions — `c1 = 1e-4`, `c2 = 0.9`,
`tolerance_grad = 1e-10`, the `1e-6` per-event convergence threshold, the per-event
scaling — and would have to change together. They have already drifted in the two
places listed above.

### Options

A. Keep both implementations and both fit drivers as they are.
B. Make `univariate` a thin wrapper over `multivariate` at `d = 1`. Loses the 1.5x on
   the value-only kernel and turns `multivariate_equivalence.rs` into a tautology.
C. Keep both evaluation kernels; move the fit driver — `Problem`, solver assembly,
   convergence measurement, evaluation counters — into one private function that
   takes the objective and gradient as closures over a flat `Vec<f64>`, and have both
   `fit`s call it with the same settings. No trait, no generic with one instantiation:
   a private function with two callers.

### Recommendation

C. It removes the coupled duplication, makes the two fits one optimizer by
construction, and keeps the independent kernel that the bit-identity oracle needs.
The 63-against-13 discrepancy must be diagnosed first, because unifying the settings
will change one of the two fits' paths and the change has to be the right way round;
the round-trip oracles and the committed `fit-d1.json` numbers are the net. Cost:
one to two days, plus a benchmark re-run for `fit-d10` and `fit-d100` if the
multivariate settings move. Issue: #52.

## 2. `rand` in the public signature of `simulate` (#42)

### Observation

`univariate::simulate(parameters, horizon, rng: &mut impl rand::Rng)`
(`univariate.rs:482-486`) and its multivariate twin (`multivariate.rs:806-810`) take
the caller's generator through `rand`'s trait, so the caller's `rand` must be the 0.9
series this crate is built against. `cargo add rand` today resolves to 0.10 and fails
on the trait bound; the README has to tell users to pin. Every future `rand` major is
a breaking change for every caller.

### Evidence

- 32 call sites in this repository construct the generator, and every one of them is
  `ChaCha8Rng::seed_from_u64(seed)` (tests, benchmark tooling, and both Python
  bindings at `hawkes-python/src/lib.rs:213,455`). No caller in the repository passes
  anything but a freshly seeded ChaCha8.
- The generators draw only `rng.random::<f64>()` (`univariate.rs:506,519`;
  `multivariate.rs:834,858,860`), three uniforms per candidate at most. Nothing from
  `rand_distr` was used; #50 removed the dependency.
- `rand` and its transitive closure are 9 of the crate's 30 dependency crates
  (`cargo tree -p hawkes-rs -e normal`).

### Options

A. Leave it; document the pin (the status quo since 0.1.0).
B. Take `&mut impl rand::RngCore` instead of `Rng`. Slightly narrower, still `rand`.
C. Take a `u64` seed and construct the generator inside: `rand_chacha` becomes an
   ordinary private dependency, `rand` leaves the public API entirely, and the Rust
   signature matches what the Python API already is (`simulate(parameters, horizon,
   seed)`). The caller loses the ability to supply a generator of their own.
D. C, plus keep the generator-taking variant under a `rand` cargo feature for the
   caller who genuinely needs to thread their own stream.

### Recommendation

C now, with D only if a caller asks. Determinism from a seed is the property every
existing user of the function relies on; a caller-supplied generator is a capability
nobody in the repository uses, and it is the whole cost of #42. The change is breaking
and belongs in 0.2 as #42 already says. Cost: half a day including the bindings, the
benchmark tooling and the quickstart; seeded outputs stay identical if the same
ChaCha8 seeding is used internally, and the seeded simulator tests will say so.
Recorded as a comment on #42 rather than a new issue.

## 3. `argmin` and the optimizer boundary

### Observation

`argmin` 0.11 is used in exactly two places, the bodies of `univariate::fit`
(`univariate.rs:602-756`) and `multivariate::fit_from` (`multivariate.rs:953-1089`),
each behind a `use` local to the function. The boundary consists of: a `Problem`
struct implementing `CostFunction` and `Gradient` over `Vec<f64>`; construction of
`MoreThuenteLineSearch` and `LBFGS`; an `Executor` run; and reading `get_best_param`
and `get_iter` from the final state. Convergence is measured by this crate afterwards,
not read from the solver (`Fit::converged`, `univariate.rs:536-546`). Nothing of
`argmin`'s types appears in the public API; the one leak is `Error::OptimizerFailed
{ message }`, whose text is `argmin`'s own `Display`.

### Evidence

- `argmin` and `argmin-math` account for 17 of the crate's 30 dependency crates
  (`cargo tree -p hawkes-rs -e normal`, with and without them), already with
  `default-features = false`.
- The pieces that would have to be rewritten to swap it are the two `Problem` impls
  and the two solver assemblies named in §1, about 60 lines per module before
  unification and about 60 lines total after it.
- The recorded gradient norms at the end of a fit straddle the `tolerance_grad(1e-10)`
  setting: `6e-12` to `4e-11` for the four `fit-d1.json` runs, `7e-10` to `4e-9` for
  `fit-d10.json`, `3.5e-6` at the cap for `fit-d100.json`. So the solver's own
  gradient stop is the exit at `d = 1` and is not at `d = 10`, and the crate decides
  for itself in both cases, with its own `1e-6` per-event threshold, whether the
  result counts as converged. That is the right design and it is what makes the
  boundary swappable: nothing downstream trusts the solver's verdict.

### Options

A. Keep `argmin`.
B. Replace it with an in-crate L-BFGS and More-Thuente line search. The line search is
   the hard part — several hundred lines with its own convergence subtleties — and it
   would need an oracle of its own; the round-trip tests would catch a wrong
   optimum but not a slow or fragile one.
C. Keep `argmin` and shrink the surface further by unifying the two boundaries (§1,
   option C), so that a future swap is one function.

### Recommendation

C, which is the same work as §1. The dependency weight is real but the boundary is
already clean, and an optimizer written here would need the verification effort this
repository reserves for the model. No separate issue; covered by #52.

## 4. The `rayon` feature

### Observation

The feature gates one function, `negative_log_likelihood_parallel`
(`multivariate.rs:1129-1214`), one test file (`multivariate_parallel.rs`) and one
probe. It parallelises the per-component work inside each distinct time — advancing
`d` states and evaluating `d` intensities — and nothing else, because the recursion
over time is sequential. `fit` never calls it.

### Evidence

`benchmarks/results/multivariate-parallel.json`, median of 5, Apple M2:

| `d` | events | sequential | parallel | parallel / sequential |
| --- | --- | --- | --- | --- |
| 2 | 65 226 | 2.8 ms | 824 ms | 290x |
| 5 | 73 521 | 3.4 ms | 1 358 ms | 397x |
| 10 | 89 286 | 5.5 ms | 1 826 ms | 331x |
| 20 | 119 566 | 13.5 ms | 3 090 ms | 228x |

The function's doc comment says it "should not be switched on" and keeps it for two
reasons: as a recorded negative result, and because the component-major accumulation
it forced is what any future parallel path needs. The second reason no longer
requires the function: the accumulation order is now a property of the sequential
code and is guarded by the bit-identity tests regardless. The first is a fact about
`d`-way parallelism at 110 flops per dispatch that the JSON, the CHANGELOG and the
verification log all record.

What the feature costs: a public function that is two to three orders of magnitude
slower than the one beside it, a second CI configuration (`ci.yml:37-41`), and a
feature flag the benchmark tooling has to mirror so that workspace builds do not
enable it by accident (`benchmarks/tooling/Cargo.toml`).

### Options

A. Keep it as is.
B. Remove the function, the feature and the test; keep the measurement where it is
   already recorded, and say in the CHANGELOG where it went.
C. Re-aim parallelism at independent realizations, which is the shape that would pay.
   Out of scope: there is no multi-realization API and CLAUDE.md §5 forbids building
   one ahead of a caller.

### Recommendation

B, in 0.2. A public entry point whose documentation is an instruction not to use it is
negative API surface. Cost: an hour; breaking only for a caller of a feature nobody
should have enabled. Issue: #53.

## 5. The shape of `Error`

### Observation

One flat `enum Error` with 13 variants (`hawkes/src/error.rs:9-108`), `thiserror`
messages, field-carrying, `PartialEq`, no `#[non_exhaustive]`. The Python bindings
match on it exhaustively on purpose, so that a new variant is a compile error there
(`hawkes-python/src/lib.rs:28-46`).

### Evidence

- The three per-event variants, `UnsortedEvents`, `EventOutsideWindow` and
  `NonFiniteEvent`, carry an `index` that in the multivariate case is the position
  **within the offending component**, and no field says which component. #46's
  `multivariate_observation_rejects_a_violation_in_any_component` had to document
  this as the contract because it is what the type can express. A caller with ten
  components and an error at "index 3" has to search.
- `NonPositiveParameter` and `NonFiniteParameter` split scalar parameters by cause,
  while `InvalidExcitation` folds negative and non-finite into one variant with a
  `row` and `column`. The asymmetry is harmless but unmotivated.
- `OptimizerFailed { message: String }` is the one variant whose content is another
  crate's wording.
- `DimensionMismatch`'s message carries ten spaces (#51).

### Options

A. Leave the shape; fix #51.
B. Add `component: Option<usize>` (or a separate `component` field set to `0` by the
   univariate path) to the three per-event variants. Breaking for anyone matching on
   them; pre-alpha.
C. Also add `#[non_exhaustive]`. This would force a wildcard arm in the bindings and
   defeat the compile-time check that every variant is mapped; the check is worth
   more than the forward compatibility while the crate is 0.x.

### Recommendation

B, without C. The component is the one piece of information a multivariate caller
needs to act on the error and cannot get. Cost: an hour, including the bindings'
message test. Issue: #54.

## 6. The Python binding layer

### Observation

`hawkes-python/src/lib.rs` (516 lines) wraps each Rust type in a `#[pyclass]`
holding the Rust value, except `Fit`: `UnivariateFit` and `MultivariateFit`
(`lib.rs:146-161`, `:332-347`) copy the seven fields of the Rust `Fit` into seven
`#[pyo3(get)]` fields each, so the same list appears four times across the two
crates. The bindings validate at the boundary only what the core cannot see: dtype
and dimensionality (by the `PyReadonlyArray` types) and squareness of the excitation
array (`lib.rs:261-265`), which the core's flat `Vec` cannot detect. #50 removed the
one check that duplicated the core.

The module docstring promises that the Python API "mirrors the Rust one and adds
nothing". It mirrors most of it: missing are `univariate::compensator_at_events`
(the multivariate one is bound), `multivariate::fit_from`, and the `Observation` types
(Python passes arrays and a horizon, which is a reasonable choice, but it is a
difference). `hawkes.__version__` is typed by hand and wrong (#51).

### Evidence

- `Fit` field lists: `univariate.rs:529-564`, `multivariate.rs:877-889`,
  `lib.rs:147-160`, `lib.rs:333-346`. Adding a field to a `Fit` is four edits and the
  compiler catches only two of them.
- Bound functions: 9 (`lib.rs:485-516`); Rust public free functions: 11, plus
  `fit_from`.

### Options

A. Leave as is.
B. Hold `inner: hawkes::univariate::Fit` in the pyclass and expose the fields through
   `#[getter]`s, as `UnivariateParameters` already does with its `inner`. Same Python
   API, one list fewer to keep in step. Zero behavioural change.
C. Bind `univariate.compensator_at_events` and `multivariate.fit_from` so the
   docstring's claim is true. Additive; both are thin.

### Recommendation

B and C together, one small PR. B is a refactor with no observable change and could
have been in #50 but was out of its stated scope; C is additive API and needs a
CHANGELOG entry. Cost: two hours. Issue: #55.

## Summary of actions

| # | Recommendation | Issue |
| --- | --- | --- |
| 1, 3 | Unify the fit driver behind one private function; diagnose the 63-vs-13 evaluation gap first | #52 |
| 2 | `simulate` takes a `u64` seed; `rand` leaves the public API | comment on #42 |
| 4 | Remove `negative_log_likelihood_parallel` and the `rayon` feature in 0.2 | #53 |
| 5 | Add the component to the three per-event error variants | #54 |
| 6 | Bindings hold the Rust `Fit`; bind the two missing functions | #55 |
| 3 | Keep `argmin` | no action |
