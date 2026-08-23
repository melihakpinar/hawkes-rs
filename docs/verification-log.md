# Verification log

Evidence that this repository's oracles detect the failures they exist to detect.

CLAUDE.md §3: *"When you add an oracle, first prove it works: deliberately break the
code it guards, confirm the test goes red, then revert. An oracle that has never gone
red is not known to be an oracle."*

This file is the record of that. Every entry below was run, and the failure output is
quoted verbatim rather than paraphrased. The working tree was restored after each,
and the fixture checksums were re-verified afterwards
(`shasum -a 256 -c`, all six `OK`).

Environment: `rustc 1.95.0`, macOS (darwin 22.6.0), `tick` oracle image
`hawk-tick:0.8.0.2`.

### Cross-machine reproducibility, confirmed

The fixtures were generated on macOS/arm64, where the pinned `linux/amd64` image runs
under emulation. CI then rebuilt that image from scratch on a native `linux/amd64`
runner, regenerated all six fixtures and compared them against the committed files
with `generate_fixtures.py --check`. All six reported `MATCH`.

Byte-identical output across two different host architectures is the property the
platform pin exists to buy, and it is now checked on every pull request rather than
asserted once.

---

## Baseline: green

```
cargo fmt --check                                          clean
cargo clippy --all-targets --all-features -- -D warnings   clean
cargo test                                                 8 passed, 0 failed
```

The eight are: three in `differential_tick`, two in `roundtrip_proptest` and three
in `gradient_check`. Two further invariants in `gradient_check` are `const`
assertions, checked by the compiler rather than the test runner, so they do not
appear in that count — see S9.

---

## Harness 1 — differential test against `tick`

`hawk/tests/differential_tick.rs`. Goal step 5.

### S1 — perturb the stub log-likelihood by `+1e-6`

The core check: does the harness actually compare, at a tolerance that matters? The
perturbation is a thousand times `LOG_LIKELIHOOD_TOLERANCE` and far too small to
notice by reading a fixture.

```rust
-    evaluation.tick_loss
+    evaluation.tick_loss + 1e-6
```

**RED**, as required:

```
test differential_against_tick ... FAILED
panicked at hawk/tests/differential_tick.rs:131:13
```

The other two tests in the file stayed green, so the failure was localized rather
than collateral. Reverted; green.

### S2 — transpose the adjacency matrix when rebuilding `coeffs`

This is CLAUDE.md §1.3's "multivariate index order" hazard, injected directly: a
transposed matrix produces plausible-looking numbers.

```rust
-            for row in &evaluation.adjacency { expected.extend_from_slice(row); }
+            for column in 0..evaluation.adjacency.len() {
+                for row in &evaluation.adjacency { expected.push(row[column]); }
+            }
```

**RED**:

```
test fixture_evaluation_coeffs_use_ticks_layout ... FAILED
panicked at hawk/tests/differential_tick.rs:270:17
```

Note what stayed green: `differential_against_tick` and
`fixtures_are_internally_consistent`. Only the test that claims to pin the layout
failed. That is the intended blast radius — and it confirms the corpus contains
asymmetric fixtures, since on a symmetric one a transpose is undetectable by
construction. Reverted; green.

### S7 — corrupt a committed fixture

Guards the corpus rather than the code. One timestamp in `univariate_tiny.json` was
moved past `end_time`.

**RED**, with a message that names the file and component:

```
test fixtures_are_internally_consistent ... FAILED
univariate_tiny: component 0 has a timestamp outside [0, end_time]
```

Fixture restored; checksum re-verified against the pre-sabotage manifest.

### S8 — remove the fixtures entirely

The failure mode that matters most for a data-driven harness: sweeping zero inputs
and reporting success. `tests/fixtures/` was replaced with an empty directory.

**RED on all three tests**, with an actionable message rather than a vacuous pass:

```
no fixtures found in .../tests/fixtures. Regenerate them with the pinned tick
image; see benchmarks/docker/README.md
```

Restored; green.

---

## Harness 2 — round-trip property test

`hawk/tests/roundtrip_proptest.rs`. Goal step 6.

### S3 — make the stub return fixed parameters

Exactly the sabotage the goal specifies for this step: an estimator that ignores its
input and returns a constant.

```rust
-fn stub_simulate_and_fit(truth: Parameters) -> Parameters { truth }
+fn stub_simulate_and_fit(_truth: Parameters) -> Parameters {
+    Parameters { baseline: 1.0, excitation: 0.5, decay: 1.0 }
+}
```

**RED**, and `proptest` shrank the counterexample to the corner of the generator's
range:

```
test simulate_then_fit_recovers_parameters ... FAILED
    baseline: 0.05,
    excitation: 0.01,
    decay: 0.1,
```

Shrinking working is itself worth confirming: in M1 it is what turns a failure into
a diagnosable one. Reverted; green.

### S4 — let the generator emit non-stationary parameters

The generator is part of the harness. If it produced parameters no Hawkes process is
defined for, every downstream failure would be ambiguous.

```rust
-    (0.05f64..5.0, 0.01f64..0.9, 0.1f64..5.0)
+    (0.05f64..5.0, 0.01f64..1.9, 0.1f64..5.0)
```

**RED**, catching a branching ratio above 1:

```
test generator_only_emits_stationary_parameters ... FAILED
    excitation: 1.5706592800173285,
```

Reverted; green.

---

## Harness 3 — finite-difference gradient check

`hawk/tests/gradient_check.rs`. Goal step 7.

Green against closed-form functions first: a quadratic and `exp(x) + y*ln(z)`, both
with gradients taken by hand. Neither is a Hawkes quantity, per the goal.

### S0 — the harness's own first design was wrong

Recorded because it is the strongest evidence in this file: this harness went red
before any sabotage, on a bug in itself.

The original version compared gradients with an **absolute** tolerance. It failed at
`(x, y) = (100, 0.5)`:

```
at [100.0, 0.5]: analytic [608.0, 201.0] vs numeric [608.0000006477349, ...],
max discrepancy 6.477348506450653e-7 > 1e-7
```

This is correct behaviour by the checker and a real defect in its design. A central
difference's round-off floor is `eps * |f| / (h * |f'|)`, which grows with the
*value* of `f`, not its derivative — so no absolute tolerance holds uniformly. Fixed
by measuring discrepancy relative to `max(1, |analytic|, |numeric|)`; the same worst
case is then `1.07e-9`, matching the predicted floor.

The tolerance was **not** loosened to make the test pass. Had the failure been
answered by relaxing 1e-7, the harness would have been silently weakened at exactly
the operating points where a gradient bug is hardest to see.

### S5 — negate one component of the analytic gradient

```rust
-vec![6.0*x + 2.0*y + 7.0,   2.0*x + 10.0*y - 4.0]
+vec![6.0*x + 2.0*y + 7.0, -(2.0*x + 10.0*y - 4.0)]
```

**RED on two tests**, which is the correct blast radius:

```
test central_difference_matches_quadratic ... FAILED
at [0.0, 0.0]: analytic [7.0, 4.0] vs numeric [7.000000000001449, -4.000000000026205],
max discrepancy 1.9999999999934488 > 1e-7

test detects_a_wrong_gradient ... FAILED
harness failed to detect a flipped sign
```

The second failure is the interesting one: `detects_a_wrong_gradient` builds its
wrong gradients *from* `quadratic_gradient`, so corrupting the source of truth makes
the "wrong" gradient accidentally right. The harness noticed. Reverted; green.

### S6 — replace central differences with forward differences

Tests that the harness's accuracy claim is real and not accidental. Forward
differences are `O(h)` rather than `O(h^2)`; at `h = 1e-5` the difference is visible.

```rust
-        gradient.push((forward - backward) / (2.0 * step));
+        gradient.push((forward - f(point)) / step);
```

**RED on both closed-form tests**:

```
test central_difference_matches_quadratic ... FAILED
max discrepancy 1.2499994483161636e-5 > 1e-7

test central_difference_matches_transcendental ... FAILED
max discrepancy 4.99998196496123e-6 > 1e-7
```

`detects_a_wrong_gradient` correctly stayed green — a less accurate numeric gradient
still detects a sign flip. Reverted; green.

### S9 — loosen `GRADIENT_TOLERANCE` to `1e-4`

The tolerance is guarded by `const _: () = assert!(...)`, so this fails at **compile
time** rather than at test time:

```
error[E0080]: evaluation panicked: GRADIENT_TOLERANCE is loose enough to admit real
derivative errors
   --> hawk/tests/gradient_check.rs:218:15
```

A future change that quietly relaxes the tolerance to make a failure go away will not
build. Reverted; green.

### Permanent detection path

`detects_a_wrong_gradient` is the sabotage made permanent: rather than relying on
this log alone, it asserts on every run that the comparator *reports* a discrepancy
for a flipped sign, transposed components, a dropped term, and a relative
perturbation of 1e-5. The detection path is therefore exercised continuously, not
only on the day it was written.

---

---

# M1 Part B

Environment as above. `hawk` now has algorithms, so from here the oracles guard real
code rather than stubs.

## Harness 4 — brute-force reference (step 5)

`hawk/tests/reference_loglikelihood.rs`. The reference every other likelihood test is
measured against, so it cannot be checked against `hawk`. Its expected values are hand
calculations written out in the tests, plus the Poisson degenerate identity.

### S10 — strict bounds relaxed to inclusive

`t_i < t_k` -> `t_i <= t_k` in the reference's inner sum. This is CLAUDE.md §1.3's
sum-bounds hazard.

**RED on three tests**: both hand calculations and the tie case. Reverted; green.

### S11 — drop the `beta` factor from the kernel, and a real gap it exposed

`alpha * beta * exp(...)` -> `alpha * exp(...)`, i.e. silently switching to
[Laub2015]'s parametrization.

**RED, but only on one test** — and that is the finding. `matches_hand_calculation_two_events`
passed, because it used `beta = 1.0`, where `alpha*beta` and `alpha` coincide. A test
that cannot distinguish the two conventions is no guard against the single most
consequential convention in this repository.

The hand calculation was recomputed with `beta = 1.5` and the test now fails under the
same sabotage (`5.396890080120764` correct, `5.422964788539137` sabotaged). The
blind spot was found by sabotage and would not have been found by reading.

## Harness 5 — the O(n) recursion (step 7)

`hawk/tests/loglikelihood.rs`. Gated against the brute force, relative to the
computation scale rather than to `|nll|`.

### S12 — use the textbook recursion [Laub2015, eq. 20]

Removed the distinct-time guard so the state advances on every event with
multiplicity 1 — exactly the published form.

**RED on `agrees_with_brute_force_on_tied_input`, green on everything else.** That is
the precise signature this bug should have: the textbook form is correct for distinct
timestamps and wrong only at ties, which is why it survives in the literature and why
the tied fixtures and tied test cases exist.

### S13 — advance the state with `1.0` instead of the multiplicity

`gap_decay * (B + count_at_previous_time)` -> `gap_decay * (B + 1.0)`. Differs from
S12 in mechanism, identical in effect: a triple tie contributes once instead of three
times.

**RED on the tied test only.** Reverted; green.

## Harness 6 — the analytic gradient (step 8)

`hawk/tests/gradient.rs`, using the same central-difference checker `gradient_check.rs`
already proved can go red.

### S14 — drop `beta * Bp_j` from (G.4)

The term the derivation singles out as the one most likely to be omitted: `lambda_j`
depends on `beta` both directly and through `B_j(beta)`.

**RED** on the randomized sweep and on the tied cases. This is the sabotage that
matters most, because `tick` cannot check `d/dbeta` at all — `decay` is a fixed
constructor argument there, not a coefficient — so this test is the only oracle for
(G.7).

### S15 — compute `Bp_j` from the pre-update state

`-gap * advanced` -> `-gap * excitation_state`, i.e. hazard 1 of the gradient
derivation §5: (G.6) requires the *advanced* value.

**RED.** Worth noting this sabotage's first attempt silently applied nothing, because
the anchor text had been reformatted by `rustfmt` and the patch did not match. The run
reported all tests green. A sabotage that fails to apply looks exactly like an oracle
that does not work, and only re-running against the real line distinguished them.
Sabotage patches must be confirmed to have landed before their result is believed.

### S16 — drop the chain-rule factor in log space (G.8)

`parameters.decay * self.decay` -> `self.decay` in `to_log_parameter_space`.

**RED on all three gradient tests**, and only in the log-space assertions — the
natural-space check is blind to it by construction, which is exactly why the
derivation requires the finite-difference check to run in both parametrizations.

## Harness 7 — the simulator and the compensator (step 6)

`hawk/tests/simulator.rs`. The two CLAUDE.md §3 oracles that could not exist before
there was a simulator.

### S17 — an accepted event does not update the excitation state

`excitation += 1.0` -> `excitation += 0.0`, which turns the simulator into a
homogeneous Poisson process at rate `mu` while leaving everything else intact.

**RED on both oracles.** The realization still looks like a point process, is still
sorted, still lies in the window, and still has no ties — the structural tests all
pass. Only the two statistical oracles notice.

### S18 — the thinning bound is wrong

Halved the bound, so it no longer dominates the intensity and thinning rejects too
much.

**RED on both oracles.** Reverted; green.

### S19 — the compensator drops its counting term

`mu*t + alpha*(m_j - B_j)` -> `mu*t + alpha*(-B_j)`, leaving the simulator untouched.

**RED on the residual test, green on the mean-intensity test.** This is the asymmetry
that justifies having oracle 2 at all. The mean intensity depends only on the
simulator, so a compensator that is wrong on its own is invisible to it. Time
rescaling checks the simulator and the compensator *against each other*, and catches
exactly the bug that the cheaper oracle cannot see.

### Negative control, permanent

`the_ks_test_rejects_residuals_from_the_wrong_parameters` is in the suite rather than
in this log: it rescales a correct realization with a deliberately wrong branching
ratio and asserts the KS statistic exceeds the critical value. A statistical test that
has never rejected is not known to have power, and unlike an exact comparison its
power is not obvious by inspection.

## Harness 8 — the fit and the `tick` identity (steps 9 and 10)

### S20 — perturb the fitted baseline by 5%

**RED** on both the round-trip property test and `the_fit_actually_optimizes`. Worth
recording what 5% means here: the round-trip tolerance is expressed in standard
errors, and on these samples a 5% shift in the baseline is tens of them. The tolerance
is generous in units of noise and unforgiving of bias, which is the intent.

### S21 — drop the per-event normalization of the objective

`scale: observation.len() as f64` -> `scale: 1.0`.

**RED** on the round-trip test. This sabotage reproduces a bug that was actually
present during development, and it is the reason `Fit::converged` measures the
gradient instead of asking the optimizer: the unnormalized objective has a log-space
gradient of order `n`, so the line search's first trial step overflows `exp`, L-BFGS
gives up after one iteration and returns its own starting point. With `converged`
defined as "stopped before the iteration cap" that state reported success, with a
result 730 nats worse than the true parameters. The fix was to optimize the
likelihood per event, which is why `tick` normalizes by the jump count too.

### S22 — drop the `D*T` term from the OQ-8 identity

**RED**, and by exactly `2000.0` on `univariate_large`, whose horizon is 2000. The
offset is not a fitted constant; it is `int_0^T sum_i 1 dt`, and the test measures it
rather than accommodating it.

## A note on sabotage technique

Two sabotage patches in M1 (S15, S22) silently failed to apply because `rustfmt` had
reformatted the anchor line, and both runs reported every test green. A sabotage that
does not land is indistinguishable from an oracle that does not work, and the
indistinguishable direction is the dangerous one — it reads as "the oracle is broken"
when the truth is "nothing was broken".

Later sabotage runs assert that the patch applied and that its anchor is unique before
believing any result.

### The M0 entries, re-run under that rule

The M0 sabotages predate the apply-assertion, so by the rule above they could not be
trusted as recorded. They were re-run. Six still have a target and all six went red
again:

| ID | target | result on re-run |
| --- | --- | --- |
| S2 | adjacency transposed in the coefficient-layout check | RED |
| S5 | one analytic gradient component negated | RED |
| S6 | forward differences instead of central | RED |
| S7 | a committed fixture corrupted | RED |
| S8 | fixtures removed entirely | RED, and loudly |
| S9 | `GRADIENT_TOLERANCE` loosened to 1e-4 | RED at compile time |

Three no longer have a target, because M1 deleted what they broke, and they are
**not** claimed as current evidence:

- **S1** perturbed the stub log-likelihood. The stub is gone; S22 supersedes it, and
  is stronger, since it breaks the real identity rather than a playback value.
- **S3** made the stub fitter return constant parameters. Superseded by S20.
- **S4** made the round-trip generator emit non-stationary parameters. That generator
  was replaced; the current test draws a stationary branching ratio by construction
  and the corresponding guard is `rejects_data_that_cannot_identify_the_parameters`.

After the full re-run the working tree was byte-identical: 27 tests green, all ten
fixture checksums unchanged.

---

# Closing the three sabotage gaps (issue #7)

M1's report identified three tests that existed but had never been shown to fail.

## Harness 9 — the gate denominator (gap a)

`hawk/tests/gate_sensitivity.rs`. `computation_scale` is the denominator of the step 7
comparison, and it was unguarded in the direction that matters: **inflating it loosens
the gate, and a looser gate does not fail — it stops catching things.** That is the one
regression class a suite cannot notice by running.

The meta-tests pin the gate's sensitivity to hand calculations, computed in a separate
implementation and written out in the test. Pinning it to a value
`computation_scale` produced would move with the bug and see nothing.

### S23 — inflate the denominator by 10x

**RED on all three tests.** `the_gate_rejects_just_above_its_sensitivity` reports that
the gate accepted an error above its documented sensitivity, which is the exact
symptom of a silently weakened gate.

### S24 — sum the signed logs instead of their magnitudes

`scale += intensity.ln().abs()` -> `scale += intensity.ln()`.

**RED on `the_denominator_sums_magnitudes_not_the_signed_sum` only.** The other two
stayed green, correctly: their case has both `log lambda` positive, so the two forms
coincide there. The discriminating case was built for this — its signed sum is
`0.0924` against `12.13` of magnitude, a factor of 131.

### S25 — "simplify" the denominator to `|nll|`

The change a future reader is most likely to make, since `|nll|` is the obvious thing
to divide by.

**RED on all three**, with a message that names the diagnosis rather than just the
numbers:

```
the gate rejected an error of 7.497173422226648e-12, which is below its documented
sensitivity of 8.330192691362942e-12. The denominator has SHRUNK -- if it was replaced
by |nll| (5.396890080120764) the boundary would sit at 5.396890080120764e-12, which is
exactly this symptom.
```

## Harness 10 — does the mean-intensity oracle earn its place? (gap b)

The question was whether `converges_to_the_stationary_mean_intensity` catches anything
`time_rescaled_residuals_are_unit_exponential` does not. Both outcomes are reported, as
required.

### S26 — branching-ratio convention changed in the simulator only

`alpha` -> `alpha/beta`, i.e. the simulator switches to [Laub2015]'s kernel while the
compensator keeps ours.

**Both oracles RED.** So this sabotage demonstrates *no* unique coverage: the residual
test alone would have caught it. Taken by itself, this is an argument for deleting the
mean-intensity test.

### S27 — the same reparametrization applied *consistently*

Simulator **and** compensator both switched to `alpha*exp(-beta t)`, so they agree with
each other. Only `stationary_mean_intensity()` still reports `mu/(1-alpha)`.

**Mean intensity RED, time rescaling GREEN.**

```
seed 1: observed mean intensity 0.799 vs analytic 1.2, relative error 0.334 > 0.05.
Under the alpha/beta branching-ratio convention the prediction would be 0.7999999999999999.
```

**Verdict: keep it.** The oracle has unique coverage, and the class it covers is
exactly the one CLAUDE.md §1.3 warns about. Time rescaling checks the simulator and the
compensator *against each other*, so it is structurally blind to any error they share —
and a convention error is precisely the kind of mistake made consistently across a
codebase rather than in one place. The mean-intensity test is the only oracle here
anchored to something outside the implementation, namely [Laub2015, eq. 6].

The failure message diagnosing the exact wrong convention (`0.7999999999999999`
matching the observed `0.799`) is not decoration; it is what turns a red test into a
located bug.

Note that S26, not S27, is the sabotage the issue asked for. Had only S26 been run the
conclusion would have been the opposite and the oracle would have been deleted. The
lesson is that "does this test have unique coverage?" is not answered by one sabotage —
it is answered by finding the sabotage that *isolates* it, and failing to find one is
not the same as proving none exists.

## Harness 11 — the simulator's structural tests (gap c)

### S28 — return the events in reverse order

**RED**, `simulated_realizations_satisfy_the_input_contract` first, naming the offending
pair:

```
UnsortedEvents { index: 1, previous_index: 0, previous: 497.8483148213864,
                 current: 497.50120108924796 }
```

The residual tests went red too, since the compensator is meaningless on unsorted
input. `converges_to_the_stationary_mean_intensity` stayed green — it counts events and
does not care about their order.

### S29 — ignore the caller's RNG and seed from entropy

**RED on `simulation_is_reproducible_from_a_seed` alone.** Every statistical oracle
stayed green, which is right: they hold in distribution and do not depend on any
particular realization. Reproducibility is the one property only this test asserts, and
it is the property that makes every other failure in this suite debuggable.

---

# Harness 12 — the two evaluation paths agree bitwise (issue #13)

`hawk/tests/bit_identical_evaluation.rs`. `negative_log_likelihood` no longer delegates
to `negative_log_likelihood_and_gradient`; it is a separate loop. The two must return
the same value **bitwise**, because there is no numerical reason for them to differ at
all, so the correct tolerance is zero.

### S30 — rewrite `-exp_m1(-x)` as the algebraically equal `1 - exp(-x)`

The "harmless refactor" a future reader is most likely to make.

**GREEN on the first attempt — the test did not catch it.** That is the finding. The
per-event difference between the two forms is around `1e-16`, and against a compensator
sum of order 1 to 100 it is below the accumulator's own ulp and disappears into the
total. Every case in the file passed.

The test was extended rather than the claim weakened.
`agree_on_events_packed_against_the_horizon` puts a handful of events within `1e-11` of
`T`, so every contribution is around `1e-12` and a `1e-16` perturbation sits five
orders above the ulp of the sum instead of below it. That regime is also exactly what
`-exp_m1` was chosen for (`univariate_loglikelihood.md` §5).

**RED after that**, deterministically:

```
1e-11 .. 1e-13 at (2, 0.9, 5): value-only path gave -2.962844630156712
(bits 0xc007b3e7e2ad37e6), value+gradient path gave -2.9628446301567126
(bits 0xc007b3e7e2ad37e7); difference 4.440892098500626e-16.
```

One bit apart. The randomized case also failed on that run but had not on the previous
one, so it catches this only sometimes; the deterministic case is the reliable guard.

### S31 — drop `alpha` from the value-only path's final combination

**RED on all six tests.** Reverted; green.

### S32 — advance the excitation state with `1.0` instead of the tie multiplicity

**RED on the two tie-heavy tests only**, green on the rest, which is the correct blast
radius for a bug that only exists at ties.

### Note on what this harness is for

A red result here does not by itself mean a wrong answer — S30's two forms are both
defensible, and differ by one ulp. It means the two paths have stopped agreeing
exactly, and either one must be brought back into line or the divergence must be made
deliberate, in the test and in the doc comment together.

---

# M2: the multivariate oracles

## Harness 13 — `d = 1` equivalence (step 9)

`hawk/tests/multivariate_equivalence.rs`. One assertion, and the goal called it the
highest-value one here: at `d = 1` the multivariate path must return **bitwise** the
same value and gradient as the univariate path.

### S34 — accumulate `count * value` instead of per event

**GREEN on the first attempt**, and that is the finding — for the second time in this
repository, after S30.

The tie cases already included multiplicity 7, chosen because Part A had shown
`7*x != x+x+...+x` for the value it tested there. It does not hold for the value this
test accumulates. A short search found that multiplicity **6** discriminates broadly
across decays and windows while 7 does not, for the compensator contributions in play.

Which multiplicities discriminate depends on the value being accumulated, and reasoning
about it case by case is exactly the sort of thing that produces a test which looks
thorough and checks nothing. Multiplicities 4, 5, 6, 7 and 9 are now all present.
**RED** after that.

### S35 — regroup the intensity product as `beta * (alpha * state)`

Algebraically identical, bitwise not.

**RED** on the packed-against-the-horizon, tie-heavy and long cases; green on the
short degenerate ones, where the rounding happens to agree.

## Harness 14 — the multivariate recursion (step 8)

`hawk/tests/multivariate_loglikelihood.rs`.

### S33 — transpose the excitation index in the intensity

`excitation[i*d + j]` -> `excitation[j*d + i]`, i.e. "i excites j" instead of "j excites
i" (`conventions.md` C6).

**RED on all eight tests.** The corpus and the hand-calculated cases are asymmetric
throughout, so there is nowhere for a transposition to hide.

### S38 — inflate the multivariate computation scale tenfold

**RED on the two gate meta-tests, green on all six comparison tests.** That split is
the whole point: a loosened gate does not fail, it stops catching things, and only a
test that pins the gate's sensitivity notices.

## Harness 15 — the multivariate gradient (step 11)

`hawk/tests/multivariate_gradient.rs`.

### S36 — drop `beta * state_derivative` from the pair accumulator

The term (G.4) warns about, which in `d` dimensions appears once per `(i, j)` pair.

**RED on all five tests.**

### S37 — transpose the excitation index in the `d/dbeta` assembly

**RED on all five.** Note this is a different transposition from S33: it is in (MG.8)'s
weighting rather than in the intensity, and the likelihood tests cannot see it at all.

### A boundary case the univariate check could not have

`alpha[i][j] = 0` is legitimate in `d` dimensions and sits on the boundary of the
domain, where a central difference would evaluate at `-h` and be rejected. The first
version of this harness crashed there. It now uses the second-order one-sided formula
`(-3f(x) + 4f(x+h) - f(x+2h)) / 2h` at zero coordinates — second-order rather than the
first-order forward difference, whose `O(h)` error would be `1e-5` and would not fit
inside the `1e-6` gate.

## Harness 16 — the multivariate simulator (step 7) and stationarity

`hawk/tests/multivariate_simulator.rs`. Both CLAUDE.md §3 external oracles, applied
**per component**. A pooled mean-intensity test would let an error that moves activity
between components cancel exactly: the total right, the process wrong. That is not
hypothetical — a transposed excitation matrix does precisely that whenever the column
sums are equal. `the_per_component_test_is_stronger_than_the_total` asserts the test
parameters have genuinely different component rates, so the per-component assertion is
a real constraint rather than a total in disguise.

### A bug the tests found before any sabotage

`branching_ratio_spectral_radius` originally returned the **midpoint** of the
Collatz-Wielandt bracket. That is wrong for reducible matrices: for
`diag(0.2, 0.7, 0.4)` the ratios `(A x)_i / x_i` equal `m_i` exactly at every step, so
the bracket is `[0.2, 0.7]` forever and the midpoint gives `0.45` where the spectral
radius is `0.7`.

Collatz-Wielandt gives `rho(A) = inf over positive x of max_i (A x)_i / x_i`, so it is
the **upper** bound that converges to `rho` for any non-negative matrix; the lower
bound converges only when `A` is irreducible. The routine now returns the upper bound
and stops when it stops moving, rather than waiting for a bracket that may never close.

The diagonal case was in `spectral_radius_matches_hand_calculations` because a diagonal
matrix has an obvious answer, not because the failure was anticipated.

### S41 — remove the spectral-radius check from `stationary_mean_intensity`

Constraint (b): invertibility of `I - alpha` does not establish stationarity.

**RED on `non_stationary_parameters_have_no_mean_intensity` alone.** For
`alpha = [[0, 2], [0.9, 0]]` the spectral radius is `1.3416` but
`det(I - alpha) = -0.8`, so with the check removed the solve succeeds and returns a
vector with negative entries. Nothing about the linear algebra failing would have
signalled the problem: the answer looks like an answer.

## Harness 17 — `tick` differential at varying `D` (step 12)

`hawk/tests/differential_tick.rs`, now comparing every fixture at `d` in
{1, 2, 3, 10}.

### S39 — hardcode `D = 3` instead of the fixture's dimension

**RED**, off by exactly `400` on a fixture with `D*T = 800` and `T = 400`. This is the
sabotage the milestone asked for by name: with a single `D` in the corpus the test
would have passed, and the offset would have been confirmed only coincidentally.

### S40 — drop the `D*T` offset entirely

**RED**, off by exactly `800` — the value of `D*T` for that fixture.

## Harness 18 — the fit and the parameter space (step 13, constraint (a))

`hawk/tests/multivariate_roundtrip.rs`. Recovery is compared **elementwise** on the
excitation matrix, because every aggregate comparison is blind to a transposition:
`transposition_is_caught_by_the_elementwise_comparison` shows that a transpose
preserves the Frobenius norm exactly and the spectral radius exactly, so a norm-based
or radius-based recovery test cannot see it.

### S42 — treat the log coordinate for `excitation` as if it were natural

Drops the `exp` in `fit`'s parameter reconstruction, i.e. the boundary conversion that
`docs/derivations/parameter_space.md` is about.

**RED on three of four tests.** The true-zero case reports the entry recovered as
`26.96` against a genuinely non-zero entry of `1.42`, which is the shape of the error
rather than just its size.

## Harness 19 — parallel and sequential agree bitwise (step 14)

`hawk/tests/multivariate_parallel.rs`, `--features rayon`.

### S43 — combine the per-component parts in reverse index order

Mathematically invisible; passes any tolerance.

**RED on `agree_over_random_shapes`, by one ulp** —
`-89.82569657461426` against `-89.82569657461427`. The two fixed-shape tests stayed
**green**: the hand-chosen cases round the same either way, and only the randomized
sweep found a shape where they do not. That is CLAUDE.md §3's fixed-seed rule again, in
a different guise.

### The measurement, recorded because it is negative

`negative_log_likelihood_parallel` is correct and bitwise identical, and it is
**228x to 397x slower** than the sequential path
(`benchmarks/results/multivariate-parallel.json`). At one distinct time the parallel
work is `O(d + d^2)` flops — 110 at `d = 10` — and thread dispatch costs far more.

It is kept, off by default, because "component-level parallelism does not pay here, by
two to three orders of magnitude" is more useful to the repository than an absence, and
because the component-major accumulation it forced is what any future parallel path
will need.

## Harness 20 — the spectral radius on reducible matrices (issue #20)

`hawk/tests/spectral_radius.rs`. The Collatz-Wielandt midpoint bug was found by a
hand-written case rather than by a sabotage, and nothing pinned the class it belonged
to. This file pins it: diagonal, block-diagonal and triangular matrices, every expected
value computed by hand.

### S44 — return the bracket midpoint instead of the upper bound

The original defect, restored.

**RED on `diagonal_matrices`, `block_diagonal_matrices` and `triangular_matrices`;
GREEN on `irreducible_matrices_are_unaffected`.** That split is the whole point: the
midpoint is correct for irreducible matrices and wrong only for reducible ones, so a
test set without reducible cases would have accepted it. `diag(0.1, 0.9)` reports
`0.45` against `0.9`.

### The second bug, found by widening the case set

The first version of this file omitted **defective** matrices — repeated eigenvalues
with too few eigenvectors — because power iteration is known to converge slowly on
them and the cases looked likely to be about tolerance rather than correctness.

Adding them anyway, per CLAUDE.md §3's rule about widening to the regime where a
method is weakest, found that
`branching_ratio_spectral_radius` returned **`1.0` for the nilpotent matrix
`[[0,1,0],[0,0,1],[0,0,0]]`, whose spectral radius is `0`**. A trivially stationary
process — a cascade that dies out after three steps — was being reported as explosive,
and `stationary_mean_intensity` would have returned `None` for it.

The cause was a "the upper bound stopped moving" early exit. On that matrix the upper
bound sits at `2` for two consecutive iterations before descending towards `1`, and the
exit fired there. It has been removed: only the bracket closing, or the iteration cap,
now stops the loop. The nilpotent case returns `2.0e-4` instead of `1.0`.

### S45 — restore that early exit

**RED on `defective_matrices` and `a_nilpotent_cascade_is_stationary` alone**, with all
five other tests green — so the regression is pinned exactly where it lives.

### Accuracy, stated rather than left to be found

Defective matrices converge sublinearly and land within about `3e-4` at the iteration
cap, against `1e-9` or better for everything else. That is immaterial for deciding
`rho < 1`, which is what the value is for, and it is now in the doc comment and pinned
by `DEFECTIVE_TOLERANCE` rather than hidden by omitting the cases.

## Summary

| ID | Harness | What was broken | Result |
| --- | --- | --- | --- |
| S0 | gradient | (found in the harness itself: absolute tolerance) | RED, design fixed |
| S1 | differential | stub log-likelihood perturbed by 1e-6 | RED |
| S2 | differential | adjacency transposed in the coefficient layout | RED |
| S3 | round-trip | estimator returns constant parameters | RED |
| S4 | round-trip | generator emits non-stationary parameters | RED |
| S5 | gradient | one analytic gradient component negated | RED |
| S6 | gradient | central differences replaced by forward differences | RED |
| S7 | differential | a committed fixture corrupted | RED |
| S8 | differential | fixtures removed entirely | RED |
| S9 | gradient | tolerance loosened to 1e-4 | RED (compile time) |
| S10 | reference | strict bounds relaxed to inclusive | RED |
| S11 | reference | kernel's `beta` factor dropped | RED, after fixing a test blind spot it exposed |
| S12 | recursion | textbook [Laub2015, eq. 20] instead of the grouped form | RED on ties only |
| S13 | recursion | state advanced with 1.0 instead of the multiplicity | RED on ties only |
| S14 | gradient | `beta*Bp_j` dropped from (G.4) | RED |
| S15 | gradient | `Bp_j` computed from the pre-update state | RED |
| S16 | gradient | chain-rule factor dropped from (G.8) | RED in log space only |
| S17 | simulator | accepted event does not excite (becomes Poisson) | RED on both oracles |
| S18 | simulator | thinning bound halved | RED on both oracles |
| S19 | compensator | counting term dropped | RED on residuals only |
| S20 | fit | fitted baseline perturbed by 5% | RED |
| S21 | fit | per-event normalization dropped | RED |
| S22 | differential | `D*T` term dropped from the OQ-8 identity | RED, by exactly `T` |
| S23 | gate denominator | `computation_scale` inflated 10x | RED |
| S24 | gate denominator | magnitudes replaced by the signed sum | RED on the mixed-sign case only |
| S25 | gate denominator | "simplified" to `|nll|` | RED, with the diagnosis named |
| S26 | simulator | branching ratio `alpha` -> `alpha/beta`, simulator only | RED on **both** oracles — no unique coverage shown |
| S27 | simulator + compensator | the same, applied consistently | mean intensity RED, time rescaling GREEN — unique coverage proven |
| S28 | simulator | events returned in reverse order | RED |
| S29 | simulator | caller's RNG ignored | RED on reproducibility alone |
| S30 | value/gradient agreement | `-exp_m1(-x)` rewritten as `1 - exp(-x)` | GREEN at first — exposed a test gap; RED after the near-horizon case was added |
| S31 | value/gradient agreement | `alpha` dropped from the value-only combination | RED |
| S32 | value/gradient agreement | state advanced without the tie multiplicity | RED on tied cases only |
| S33 | multivariate likelihood | excitation index transposed in the intensity | RED, all eight |
| S34 | `d = 1` equivalence | `count * value` instead of per-event accumulation | GREEN at first — exposed a test gap; RED once multiplicity 6 was added |
| S35 | `d = 1` equivalence | intensity product regrouped | RED |
| S36 | multivariate gradient | `beta*Bp` dropped from the pair accumulator | RED |
| S37 | multivariate gradient | excitation index transposed in `d/dbeta` | RED |
| S38 | multivariate gate | computation scale inflated 10x | RED on the meta-tests only |
| S39 | `tick` differential | `D` hardcoded to 3 | RED, off by exactly `|D-3|*T` |
| S40 | `tick` differential | `D*T` offset dropped | RED, off by exactly `D*T` |
| S41 | stationarity | spectral-radius check removed | RED on the non-stationary case alone |
| S42 | fit parameter space | log coordinate for `excitation` treated as natural | RED on 3 of 4 |
| S43 | parallel path | per-component parts combined in reverse order | RED on the randomized case, by one ulp; fixed shapes stayed green |
| S44 | spectral radius | bracket midpoint instead of the upper bound | RED on the reducible cases, GREEN on irreducible |
| S45 | spectral radius | "upper bound stopped moving" early exit restored | RED on the defective cases alone |

Every harness has been observed both red and green. The working tree after all
sabotages is byte-identical to before them.

## Not yet proven

Two of CLAUDE.md §3's five oracles do not exist yet and are not claimed here:

1. **Analytic identity** (stationary mean intensity) — needs a simulator. M1.
2. **Time-rescaling / Ogata residuals** — needs a simulator and a compensator. M1.

The three that do exist are the three that can be built without algorithm code.
