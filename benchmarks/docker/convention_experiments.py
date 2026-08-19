"""Experiments that settle the convention questions in docs/open-questions.md.

Run inside the pinned oracle image:

    docker run --rm --platform=linux/amd64 hawk-tick:0.8.0.2 \
        python /work/convention_experiments.py

Everything printed here is evidence cited by docs/derivations/conventions.md. The
output is deterministic: no simulation, no seeds, only fixed event lists.

# Why the loss is not used as evidence

`ModelHawkesExpKernLogLik.loss` carries an offset that OQ-8 has only confirmed at
`adjacency == 0`. Using an absolute loss value to settle the kernel form or the sum
bounds would assume the very thing OQ-8 leaves open. Two routes avoid that:

* `HawkesKernelExp` evaluates the kernel directly, with no likelihood involved.
* `model.grad` is the derivative of whatever `loss` computes, so **any**
  parameter-independent offset differentiates away. E1c checks that tick's gradient
  really is the derivative of tick's loss, which is what licenses this route.

Loss values are compared only between two runs sharing the same parameters, D, T and
n_jumps, where the offset is identical on both sides and cancels in the difference.
"""

import numpy as np
from tick.hawkes import HawkesKernelExp, ModelHawkesExpKernLogLik

ALPHA, BETA, MU = 0.35, 1.3, 0.7
EVENTS = [1.0, 2.0, 2.5]
END_TIME = 3.0


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def grad_of(events, end_time, mu, alpha, decay=BETA, n_nodes=1):
    model = ModelHawkesExpKernLogLik(decay=decay)
    model.fit([[np.asarray(e, dtype=float) for e in events]], end_times=end_time)
    coeffs = np.concatenate([np.full(n_nodes, mu), np.full(n_nodes * n_nodes, alpha)])
    grad = model.grad(coeffs, out=np.zeros(coeffs.size))
    return model, coeffs, grad


# ---------------------------------------------------------------------- E1: OQ-1
rule("E1 — kernel normalization (OQ-1) and branching ratio (OQ-2)")

kernel = HawkesKernelExp(ALPHA, BETA)
print(f"HawkesKernelExp(intensity={ALPHA}, decay={BETA})")
print(f"  H1: alpha*beta*exp(-beta t)   H2: alpha*exp(-beta t)")
for t in (0.0, 0.5, 1.0, 2.0):
    measured = kernel.get_value(t)
    h1 = ALPHA * BETA * np.exp(-BETA * t)
    h2 = ALPHA * np.exp(-BETA * t)
    print(f"  t={t}: measured={measured!r}  H1={h1!r} {'MATCH' if close(measured, h1) else ''}"
          f"  H2={h2!r} {'MATCH' if close(measured, h2) else ''}")

norm = kernel.get_norm()
print(f"\n  get_norm()={norm!r}   alpha={ALPHA} {'MATCH' if close(norm, ALPHA) else ''}"
      f"   alpha/beta={ALPHA / BETA!r} {'MATCH' if close(norm, ALPHA / BETA) else ''}")
print("  get_norm() is the kernel integral, so it is the branching ratio (OQ-2).")

print("\n  primitive (kernel integral from 0 to t), for the compensator:")
for t in (0.5, 1.0, 3.0):
    prim = kernel.get_primitive_value(t)
    h1 = ALPHA * (1.0 - np.exp(-BETA * t))
    print(f"    t={t}: measured={prim!r}  alpha*(1-exp(-beta t))={h1!r} "
          f"{'MATCH' if close(prim, h1) else 'DIFFER'}")

# ------------------------------------------------------------- E2: OQ-1 and OQ-3
rule("E2 — kernel form and sum bounds, via tick's gradient (OQ-1, OQ-3)")

print("grad is the derivative of whatever loss tick computes, so a")
print("parameter-independent offset cannot influence it.\n")
print("  d(loss*n)/dmu    = T - sum_k 1/lambda(t_k)")
print("  d(loss*n)/dalpha = sum_i dPhi(T-t_i)/dalpha")
print("                     - sum_k (1/lambda(t_k)) dlambda(t_k)/dalpha\n")

model, coeffs, grad = grad_of([EVENTS], END_TIME, MU, ALPHA)
n_jumps = len(EVENTS)
measured_mu, measured_alpha = grad[0] * n_jumps, grad[1] * n_jumps
print(f"events={EVENTS} T={END_TIME} mu={MU} alpha={ALPHA} beta={BETA} n={n_jumps}")
print(f"  tick grad*n: d/dmu={measured_mu!r}  d/dalpha={measured_alpha!r}\n")

ev = np.array(EVENTS)
for kernel_name, phi, dphi_dalpha, prim_dalpha in [
    ("H1 alpha*beta*exp", lambda s: ALPHA * BETA * np.exp(-BETA * s),
     lambda s: BETA * np.exp(-BETA * s), lambda s: 1.0 - np.exp(-BETA * s)),
    ("H2 alpha*exp", lambda s: ALPHA * np.exp(-BETA * s),
     lambda s: np.exp(-BETA * s), lambda s: (1.0 - np.exp(-BETA * s)) / BETA),
]:
    for bound_name, strict in [("strict t_i<t_k", True), ("inclusive t_i<=t_k", False)]:
        lam, dlam = [], []
        for k, tk in enumerate(ev):
            prior = ev[ev < tk] if strict else ev[(ev <= tk) & (np.arange(len(ev)) != k)]
            # "inclusive" means an earlier-or-simultaneous event contributes; an
            # event never excites itself, so index k is excluded either way.
            lam.append(MU + phi(tk - prior).sum())
            dlam.append(dphi_dalpha(tk - prior).sum())
        lam, dlam = np.array(lam), np.array(dlam)
        pred_mu = END_TIME - (1.0 / lam).sum()
        pred_alpha = prim_dalpha(END_TIME - ev).sum() - (dlam / lam).sum()
        verdict = ("MATCH" if close(pred_mu, measured_mu, 1e-9)
                   and close(pred_alpha, measured_alpha, 1e-9) else "")
        print(f"  {kernel_name:18s} {bound_name:20s} "
              f"d/dmu={pred_mu!r:22s} d/dalpha={pred_alpha!r:22s} {verdict}")

rule("E1c — is tick's grad really the derivative of tick's loss?")
h = 1e-6
for index, name in ((0, "mu"), (1, "alpha")):
    up, down = coeffs.copy(), coeffs.copy()
    up[index] += h
    down[index] -= h
    central = (model.loss(up) - model.loss(down)) / (2 * h)
    print(f"  d/d{name}: analytic={grad[index]!r} central={central!r} "
          f"{'CONSISTENT' if close(grad[index], central, 1e-6) else 'INCONSISTENT'}")


# --------------------------------------------------------------- E2b: OQ-3 proper
rule("E2b — does an event's own jump enter its intensity? (OQ-3)")

print("With distinct timestamps, 't_i < t_k' and 't_i <= t_k excluding i==k' select")
print("the same set, so E2 could not separate them. The choice with observable")
print("content is whether the jump AT t_k contributes phi(0) = alpha*beta to")
print("lambda(t_k). Only the predictable (strict) reading excludes it.\n")

for label, include_self in [("strict, predictable", False), ("self-inclusive", True)]:
    lam, dlam = [], []
    for k, tk in enumerate(ev):
        prior = ev[ev < tk]
        base = MU + (ALPHA * BETA * np.exp(-BETA * (tk - prior))).sum()
        dbase = (BETA * np.exp(-BETA * (tk - prior))).sum()
        if include_self:
            base += ALPHA * BETA          # phi(0)
            dbase += BETA
        lam.append(base)
        dlam.append(dbase)
    lam, dlam = np.array(lam), np.array(dlam)
    pred_mu = END_TIME - (1.0 / lam).sum()
    pred_alpha = (1.0 - np.exp(-BETA * (END_TIME - ev))).sum() - (dlam / lam).sum()
    verdict = ("MATCH" if close(pred_mu, measured_mu, 1e-9)
               and close(pred_alpha, measured_alpha, 1e-9) else "")
    print(f"  {label:22s} d/dmu={pred_mu!r:22s} d/dalpha={pred_alpha!r:22s} {verdict}")

# ---------------------------------------------------------------------- E3: OQ-7
rule("E3 — ordering and exact ties (OQ-7)")

print("E3a — within-component order. Same multiset, different array order.")
print("      Same parameters, D, T and n_jumps on both sides, so any loss offset")
print("      is identical and cancels in the comparison.")
sorted_events = [1.0, 2.0, 2.5]
for label, arr in [("sorted    ", sorted_events),
                   ("reversed  ", [2.5, 2.0, 1.0]),
                   ("shuffled  ", [2.0, 1.0, 2.5])]:
    try:
        m, c, g = grad_of([arr], END_TIME, MU, ALPHA)
        print(f"  {label} {arr}: loss={m.loss(c)!r} grad*n=({g[0]*3!r}, {g[1]*3!r})")
    except Exception as exc:
        print(f"  {label} {arr}: {type(exc).__name__}: {str(exc)[:70]}")

print("\nE3b — exact tie within one component: events [1.0, 2.0, 2.0].")
tied = [1.0, 2.0, 2.0]
try:
    m, c, g = grad_of([tied], END_TIME, MU, ALPHA)
    measured = g[0] * len(tied)
    print(f"  accepted. loss={m.loss(c)!r}  grad_mu*n={measured!r}")
    a = np.array(tied)
    for label, rule_fn in [
        ("tied events do NOT excite each other (strict)", lambda tk, i: a[a < tk]),
        ("a tied earlier-indexed event DOES excite", 
         lambda tk, i: a[(a < tk) | ((a == tk) & (np.arange(len(a)) < i))]),
    ]:
        lam = np.array([MU + (ALPHA * BETA * np.exp(-BETA * (tk - rule_fn(tk, i)))).sum()
                        for i, tk in enumerate(a)])
        pred = END_TIME - (1.0 / lam).sum()
        print(f"    {label:46s} d/dmu={pred!r} "
              f"{'MATCH' if close(pred, measured, 1e-9) else ''}")
except Exception as exc:
    print(f"  REJECTED: {type(exc).__name__}: {str(exc)[:100]}")

print("\nE3c — tie across two components: [[1.0], [1.0]].")
try:
    m, c, g = grad_of([[1.0], [1.0]], END_TIME, MU, ALPHA, n_nodes=2)
    print(f"  accepted. loss={m.loss(c)!r}")
except Exception as exc:
    print(f"  REJECTED: {type(exc).__name__}: {str(exc)[:100]}")

print("\nE3d — boundary timestamps: an event at exactly 0.0, and at exactly T.")
for label, arr, T in [("t=0.0 present", [0.0, 1.0, 2.0], 3.0),
                      ("t=T present  ", [1.0, 2.0, 3.0], 3.0),
                      ("t>T present  ", [1.0, 2.0, 4.0], 3.0)]:
    try:
        m, c, g = grad_of([arr], T, MU, ALPHA)
        print(f"  {label} {arr} T={T}: loss={m.loss(c)!r}")
    except Exception as exc:
        print(f"  {label} {arr} T={T}: {type(exc).__name__}: {str(exc)[:70]}")

print("\nE3e — empty component in a bivariate process.")
try:
    m, c, g = grad_of([[1.0, 2.0], []], END_TIME, MU, ALPHA, n_nodes=2)
    print(f"  accepted. loss={m.loss(c)!r}")
except Exception as exc:
    print(f"  REJECTED: {type(exc).__name__}: {str(exc)[:100]}")
