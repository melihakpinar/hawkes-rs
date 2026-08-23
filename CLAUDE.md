# CLAUDE.md — hawk

A Rust library for multivariate Hawkes processes: simulation and maximum-likelihood
estimation. Python bindings via PyO3.

`tick` remains the incumbent, is maintained, and is faster on the univariate
exponential-kernel fit — roughly 3x at n = 1e6, largely because it holds beta fixed and
can therefore precompute its exponential terms. `hawk` exists for different reasons,
each of them measured: no Rust crate does Hawkes estimation at all; `tick`'s learner
cannot express an observation window, so its baseline estimates are biased whenever the
window has trailing dead time (OQ-5); its loss is neither the log-likelihood nor its own
documented formula (OQ-8); its documented likelihood-fitting interface does not work;
and it cannot estimate beta at all. Speed is not this library's claim, and the
benchmarks must say so plainly.

The library's only real product is **correct numbers**. Speed is secondary. A fast
library that returns subtly wrong parameter estimates is worthless and worse than
nothing, because users cannot tell.

---

## 1. Prime directive: assume nothing

This is the single most important rule in this repository. It overrides convenience,
speed, and completeness.

**Never implement a formula, convention, index range, or numerical choice that you
cannot trace to a named source.** A source is:

1. A paper in `docs/references/`, cited by **equation number** (not just title).
2. `tick`'s source code, cited by **file path and line range**.
3. A derivation in `docs/derivations/` that you wrote and a human approved.
4. A property test that empirically pins the behaviour (e.g. a differential test
   against `tick`).

Nothing else counts. Not "standard practice". Not "usually people do X". Not your
prior.

### The forbidden comment

If you ever find yourself about to write a comment of this shape:

```rust
// Assuming the kernel is normalized here
// This is probably the convention used by ...
// Note: we take the sum over j < i (I think)
```

**Stop.** That comment is not documentation, it is a bug report you are filing
against yourself. The correct action is never to write the comment and proceed. The
correct action is one of:

- Go find the source that settles it, cite it, and delete the hedge.
- If no source settles it, do not implement it. Follow §1.2.

Comments explaining *why* a cited choice was made are good and encouraged. Comments
that hedge about *what* the right choice is are forbidden.

### 1.2 When a source does not settle it

The repository owner is not a domain expert and **cannot answer questions about the
mathematics**. Do not block on asking them. Instead:

1. Append an entry to `docs/open-questions.md` with: the question, exactly what you
   searched and read, the candidate resolutions, and what observably differs between
   them (e.g. "resolution A makes the log-likelihood differ from tick by a constant
   `n*ln(beta)`").
2. If a differential test against `tick` can distinguish the candidates, **write that
   test and let it decide**. This is the preferred resolution path — it turns a
   convention question into an empirical one. Record the answer and the test in the
   open-questions entry, then close it.
3. If nothing can decide it, mark the work item BLOCKED in the entry, stop work on
   it, and move to an unblocked item. Do not implement a guess and do not implement a
   configurable switch to dodge the question.

### 1.3 Known convention hazards

These specific points have more than one convention in the literature. Each one MUST
be pinned to a source before any code that touches it is written. Record the
resolution in `docs/derivations/conventions.md`.

- **Kernel normalization.** Is the exponential kernel `alpha * exp(-beta*t)` or
  `alpha * beta * exp(-beta*t)`? These give different meanings to `alpha` and a
  different branching ratio, and both appear in the literature.
- **Branching ratio.** Under the chosen normalization, is it `alpha/beta` or `alpha`?
  Stationarity requires it to be `< 1` (univariate) or spectral radius `< 1`
  (multivariate).
- **Sum bounds in the intensity.** Strict `t_j < t` or inclusive `t_j <= t`? Only one
  keeps the intensity predictable. What is the intensity at the very first event?
- **Compensator on the tail.** Does the integral term run to `T` (the observation
  horizon) or to the last event time? These differ and the difference is not constant.
- **Observation window.** Is `[0, T]` given by the caller, or inferred as
  `[0, max(events)]`? An inferred window silently biases `mu` downward.
- **Multivariate index order.** Does `alpha[i][j]` mean "j excites i" or "i excites
  j"? Getting this transposed produces a fitted matrix that looks plausible and is
  wrong.
- **Event ordering across dimensions.** Are timestamps required globally sorted, or
  sorted within each component? What happens on exact ties?

---

## 2. Sources of truth

Citations must resolve to an equation in a freely accessible source. [Laub2015, eq. N] is the primary citable reference. The original papers (Hawkes 1971, Ozaki 1979, Ogata 1981) are cited alongside for provenance, without equation numbers unless a PDF is in docs/references/.

- **Ozaki, T. (1979).** Maximum likelihood estimation of Hawkes' self-exciting point
  processes. *Ann. Inst. Statist. Math.* 31(1), 145–155. — the O(n) recursive
  likelihood. This is the core of the estimator.
- **Hawkes, A. G. (1971).** Spectra of some self-exciting and mutually exciting point
  processes. *Biometrika* 58(1), 83–90. — definition, stationarity.
- **Hawkes, A. G. & Oakes, D. (1974).** A cluster process representation of a
  self-exciting process. *J. Appl. Probab.* 11(3), 493–503. — branching interpretation.
- **Ogata, Y. (1981).** On Lewis' simulation method for point processes. *IEEE Trans.
  Inf. Theory* 27(1), 23–31. — thinning simulation.
- **Ogata, Y. (1988).** Statistical models for earthquake occurrences and residual
  analysis for point processes. *JASA* 83(401), 9–27. — time-rescaling residual
  analysis, used as a verification oracle.
- **Dassios, A. & Zhao, H. (2013).** Exact simulation of Hawkes process with
  exponentially decaying intensity. *Electron. Commun. Probab.* 18(62), 1–13.
- **Bacry, E., Mastromatteo, I. & Muzy, J.-F. (2015).** Hawkes processes in finance.
  *Market Microstructure and Liquidity* 1(1), 1550005. — multivariate setup, finance
  conventions.
- **Bacry, E., Bompaire, M., Gaïffas, S. & Muzy, J.-F. (2020).** Sparse and low-rank
  multivariate Hawkes processes. *JMLR* 21(50), 1–32. — what `tick` actually implements.
- **Bacry, E. et al. (2018).** tick: a Python library for statistical learning...
  *JMLR* 18(214), 1–5. — the incumbent, and the benchmark baseline.

`tick`'s source is the tiebreaker for conventions, because differential tests against
it are the strongest oracle available. Cite it by path and line.

---

## 3. Verification comes before features

No estimator lands without an oracle that could have caught it being wrong. These
five exist; use them. Adding a feature means adding to them.

1. **Analytic identity.** Stationary univariate mean intensity has a closed form in
   terms of `mu` and the branching ratio. Long simulations must converge to it.
2. **Time-rescaling (Ogata residuals).** Transforming event times by the compensator
   must yield a unit-rate Poisson process. KS test. This validates the simulator and
   the intensity computation *jointly* — they cannot both be wrong consistently.
3. **Round-trip property test.** `proptest`: random valid parameters → simulate →
   fit → recover parameters within a stated tolerance. This is the main regression net.
4. **Gradient check.** Analytic gradient vs central finite differences. A wrong
   derivative still converges — to the wrong place. Only this test catches it.
5. **Differential test against tick.** Same events, same parameters, log-likelihood
   agreeing to `1e-9`. Fixtures are committed in `tests/fixtures/`.

**Sabotage rule.** When you add an oracle, first prove it works: deliberately break
the code it guards, confirm the test goes red, then revert. An oracle that has never
gone red is not known to be an oracle. Note the sabotage in the test's doc comment.

**Fixed-seed single cases cannot see size-dependent defects.** A test pinned to one
seed exercises one realization; a defect whose trigger scales with `n` — gradient
magnitude, accumulated error, overflow in a line search — is structurally invisible to
it. Randomized sweeps are not a nicety here, they are the only thing that covers this
class. Every single-case test must be accompanied by a randomized one over the same
code path.

**Do not write a test whose expected value you computed with the code under test.**
Expected values come from a paper, from `tick`, or from an independent hand
calculation recorded in `docs/derivations/`.

---

## 4. Workflow

**Derivation before code.** Before implementing any formula, write the derivation to
`docs/derivations/<name>.md`: the mathematics, the index conventions, the source
citations, and the exact expression you intend to code. Then **stop and hand it to
the human for approval.** Do not proceed to implementation in the same turn. An index
error is invisible in code and obvious in a derivation — this is the whole point.

**Definition of done** for any work item:
- `cargo fmt --check` clean
- `cargo clippy --all-targets --all-features -- -D warnings` clean
- `cargo test` green
- Every new formula traces to a citation
- `docs/open-questions.md` has no new unresolved entries introduced by this work
- Public API change → `CHANGELOG.md` entry

**Never** commit a `#[ignore]`d or commented-out test to make the suite pass. If a
test fails and you cannot fix it, leave it failing and say so.

---

## 5. Code style

Rust community conventions win wherever they conflict with anything below. Run
`clippy` and believe it. The points below are where the owner has a stated preference.

**Naming.** Long names are welcome when they buy clarity.
`branching_ratio_spectral_radius` is better than `sr`. Do not abbreviate domain terms.
Single letters are acceptable *only* where they mirror a cited equation — and when you
do that, the doc comment must say which equation, e.g. `/// `mu` in [Ozaki1979, eq. 3]`.

The one place Rust convention overrides this: do not stutter across module paths.
Inside `hawkes::`, name the type `Process`, not `HawkesProcess` — clippy's
`module_name_repetitions` will tell you.

**Duplication.** Duplication is not a defect by itself. A two-line expression written
in three places is fine and often clearer than an abstraction. Extract a function when
the copies are **coupled** — when they depend on the same value and would all have to
change together if that value's meaning changed. That coupling, not the character
count, is the trigger. The Rust community broadly agrees with this ("prefer
duplication over the wrong abstraction"), so there is no conflict to resolve here.

**Do not build abstractions for a single caller.** No trait with one implementor, no
generic with one instantiation, no config struct with one field. v0.1 has one kernel
family; write it concretely. Generalize when the second kernel actually arrives.

**Errors.** Library code returns `Result` with `thiserror` types. No `unwrap`,
`expect`, or `panic!` in `src/` outside of tests and documented invariant assertions.
Invalid input is an error value, not a panic.

---

## 6. Numerical conventions

- `f64` everywhere. No `f32`, no generics over float type.
- Log-likelihood is accumulated in log space; use `ln_1p` and `exp_m1` where the
  argument is near zero.
- Positivity constraints (`mu, alpha, beta > 0`) are handled by optimizing in
  log-parameter space, not by constrained optimization. Convert at the boundary.
- Stationarity is **not** enforced during optimization. Check it after fitting and
  return it as a diagnostic on the fit result. A non-stationary fit is a real finding
  about the data, not an error.
- Never compare floats with `==`. Every tolerance is a named constant with a comment
  explaining where the number came from.

---

## 7. Layout

```
hawk/                  Rust core crate
hawk-python/           PyO3 bindings, maturin
docs/
  references/          papers (PDF)
  derivations/         approved derivations, conventions.md
  open-questions.md    unresolved items, BLOCKED work
benchmarks/
  docker/              pinned tick environment
  suite/               one runnable script per benchmark
  results/             committed JSON
tests/fixtures/        committed reference data from tick
```

---

## 8. Out of scope for v0.1.0

Do not build these, do not add hooks "in preparation" for them: sum-of-exponentials
kernels, power-law kernels, non-parametric estimation (NPHC, Wiener-Hopf), L1/nuclear
regularization, marked processes, spatial processes, GPU, async.

v0.1.0 is: univariate and multivariate exponential-kernel Hawkes, simulation, MLE,
Python wheels, benchmarks against tick. That is all.

---

## 9. Version control (non-negotiable)

These rules never change and are not subject to per-task judgement.

### Attribution

**Never attribute yourself in anything that reaches GitHub.** Specifically:

- No `Co-Authored-By: Claude` trailer, or any co-author trailer.
- No "Generated with Claude Code", "🤖", or similar footer in commit messages,
  PR titles, PR bodies, issue bodies, or review comments.
- Never modify `user.name` or `user.email`. Commits are authored by the
  repository owner's existing git configuration, unchanged.

If a tool or template would add such a footer automatically, strip it before
committing. This applies to every git and `gh` invocation without exception.

### Commit granularity

Commit every meaningful change. A meaningful change is one that leaves the
repository in a coherent state and has a single reason to exist: one derivation
approved, one oracle added, one bug fixed, one dependency pinned. Do not batch
unrelated work into one commit, and do not leave completed work uncommitted at
the end of a work session.

Never commit with a red test suite unless the commit's stated purpose is to
record a failing test, and the message says so.

### Commit messages

Conventional Commits, matching the branch prefixes:

<type>(<scope>): <imperative summary, lower case, no trailing period>

<body: why this change, not what. Cite sources where the change encodes a
mathematical decision, e.g. [Ozaki1979, eq. 7] or tick path:lines.>

Refs #<issue>


Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`, `ci`.
Scopes are crate or module names: `hawk`, `hawk-python`, `bench`, `fixtures`.

Summary line under 72 characters. Body wrapped at 72. A commit that encodes a
convention decision from CLAUDE.md §1.3 MUST cite the source that settled it in
the body — the commit log is part of the audit trail.

### Branches

Never commit directly to `main`. All work happens on a branch:

feat/<issue-number>-<short-kebab-description>
fix/<issue-number>-<short-kebab-description>


Examples: `feat/12-add-multivariate-loglikelihood`,
`fix/31-compensator-tail-integral`.

The number is the GitHub issue number. If no issue exists for the work, create
one first with `gh issue create` describing the work and its acceptance
criteria, then branch from it. Work is never branchless and never issueless.

### Pull requests

Use the `gh` CLI to open and merge PRs. A PR may be merged only when:

- CI is green (`fmt`, `clippy -D warnings`, `test`)
- The work item's own verification criteria pass
- No new unresolved entry was added to `docs/open-questions.md` by this work

PR body states: what changed, which sources back any mathematical decision, and
how it was verified. It closes its issue (`Closes #<n>`). No self-attribution
footer, per the attribution rules above.

If a PR cannot be merged because something is BLOCKED, leave it open, say why in
a comment, and move to other work. Do not merge around a blocker.
