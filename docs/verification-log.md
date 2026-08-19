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
believing any result. Any sabotage recorded in this file without that confirmation
should be re-run before it is trusted.

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

Every harness has been observed both red and green. The working tree after all
sabotages is byte-identical to before them.

## Not yet proven

Two of CLAUDE.md §3's five oracles do not exist yet and are not claimed here:

1. **Analytic identity** (stationary mean intensity) — needs a simulator. M1.
2. **Time-rescaling / Ogata residuals** — needs a simulator and a compensator. M1.

The three that do exist are the three that can be built without algorithm code.
