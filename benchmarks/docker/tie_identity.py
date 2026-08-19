"""Does the OQ-8 identity nll == tick_loss*n_jumps + D*T survive exact ties?

If `tick` resolved ties by array index -- the textbook recursion, Laub2015 eq. 20 --
the identity would have to break on tied data, and that break would not be a defect.
This decides it by evaluating the negative log-likelihood under BOTH semantics and
seeing which one `tick` agrees with.

Evidence for conventions.md C8 and open-questions.md OQ-8.

    docker run --rm --platform=linux/amd64 -v "$PWD/benchmarks/docker":/w \
        hawk-tick:0.8.0.2 python /w/tie_identity.py
"""
import numpy as np
from tick.hawkes import ModelHawkesExpKernLogLik

def nll_strict(events, mu, alpha, decay, T):
    """Grouped/strict semantics: an event never excites a simultaneous one."""
    D = len(events)
    comp = 0.0
    for i in range(D):
        comp += mu[i] * T
        for j in range(D):
            for t in events[j]:
                comp += alpha[i][j] * (1.0 - np.exp(-decay * (T - t)))
    log_term = 0.0
    for i in range(D):
        for tk in events[i]:
            lam = mu[i]
            for j in range(D):
                for t in events[j]:
                    if t < tk:                                  # strict on TIME
                        lam += alpha[i][j] * decay * np.exp(-decay * (tk - t))
            log_term += np.log(lam)
    return comp - log_term

def nll_index(events, mu, alpha, decay, T):
    """Index-based semantics (textbook 4.2): an earlier-indexed tied event DOES excite.
    Only differs from nll_strict when ties are present. Univariate only."""
    assert len(events) == 1
    t = events[0]
    comp = mu[0] * T + sum(alpha[0][0] * (1.0 - np.exp(-decay * (T - x))) for x in t)
    log_term = 0.0
    for k, tk in enumerate(t):
        lam = mu[0] + sum(alpha[0][0] * decay * np.exp(-decay * (tk - t[j]))
                          for j in range(k))                    # strict on INDEX
        log_term += np.log(lam)
    return comp - log_term

CASES = [
    ("no ties          ", [np.array([1.0, 2.0, 3.5, 4.0])], 1),
    ("one tied pair    ", [np.array([1.0, 2.0, 2.0, 3.5])], 1),
    ("triple tie       ", [np.array([1.0, 2.0, 2.0, 2.0, 3.5])], 1),
    ("ties at both ends", [np.array([1.0, 1.0, 2.0, 3.5, 3.5])], 1),
    ("bivariate, cross-component tie",
     [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])], 2),
]
T, decay = 6.0, 1.3
print(f"{'case':32s} {'D*T':>6s} {'delta(strict)':>16s} {'delta(index)':>16s}")
print("-" * 76)
for name, events, D in CASES:
    mu = [0.6] * D
    alpha = [[0.25 if i == j else 0.15 for j in range(D)] for i in range(D)]
    n_jumps = sum(len(e) for e in events)
    m = ModelHawkesExpKernLogLik(decay=decay)
    m.fit([events], end_times=T)
    coeffs = np.concatenate([np.array(mu), np.array(alpha).ravel()])
    predicted = m.loss(coeffs) * n_jumps + D * T
    d_strict = predicted - nll_strict(events, mu, alpha, decay, T)
    d_index = (predicted - nll_index(events, mu, alpha, decay, T)) if D == 1 else float('nan')
    print(f"{name:32s} {D*T:6.1f} {d_strict:16.3e} {d_index:16.3e}")
print("-" * 76)
print("delta(strict) ~ 0  => tick uses TIME-based (grouped) semantics on ties")
print("delta(index)  ~ 0  => tick uses INDEX-based (textbook 4.2) semantics on ties")
