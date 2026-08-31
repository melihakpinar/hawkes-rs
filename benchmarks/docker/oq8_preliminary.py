"""Preliminary evidence on OQ-8, run against the committed fixtures.

OQ-8 asks whether tick's loss offset stays exactly -D*T once adjacency != 0. M0
confirmed it only at adjacency == 0, where the model collapses to Poisson.

This script evaluates the multivariate negative log-likelihood directly from the
definition — brute force, no recursion — and compares:

    tick_loss * n_jumps + D*T   ==?   nll

It deliberately does NOT use `hawkes`: nothing is implemented yet, and M1 Part B step 9
closes OQ-8 properly using hawkes's own implementation after that implementation has
been validated against non-tick oracles. This is evidence gathered ahead of that, to
find out early whether candidate (a) or (b) is heading for confirmation.

    docker run --rm --platform=linux/amd64 \
        -v "$PWD/tests/fixtures":/fx -v "$PWD/benchmarks/docker":/w \
        hawkes-tick:0.8.0.2 python /w/oq8_preliminary.py
"""

import json
import pathlib

import numpy as np
from tick.hawkes import ModelHawkesExpKernLogLik


def negative_log_likelihood(events, baseline, adjacency, decay, end_time):
    """Direct transcription of the definition. No recursion, no simplification.

    lambda_i(t)  = mu_i + sum_j sum_{t_k^j < t} alpha_ij * beta * exp(-beta*(t - t_k^j))
    Lambda_i(T)  = mu_i*T + sum_j sum_{t_k^j} alpha_ij * (1 - exp(-beta*(T - t_k^j)))
    nll          = sum_i Lambda_i(T) - sum_i sum_k log lambda_i(t_k^i)
    """
    n_nodes = len(events)
    compensator = 0.0
    for i in range(n_nodes):
        compensator += baseline[i] * end_time
        for j in range(n_nodes):
            for t in events[j]:
                compensator += adjacency[i][j] * (1.0 - np.exp(-decay * (end_time - t)))

    log_term = 0.0
    for i in range(n_nodes):
        for t_k in events[i]:
            intensity = baseline[i]
            for j in range(n_nodes):
                for t in events[j]:
                    if t < t_k:                      # strict, predictable (OQ-3)
                        intensity += adjacency[i][j] * decay * np.exp(-decay * (t_k - t))
            log_term += np.log(intensity)

    return compensator - log_term


print(f"{'fixture':24s} {'point':22s} {'D*T':>8s} {'delta':>24s}  verdict")
print("-" * 96)

all_match = True
for path in sorted(pathlib.Path("/fx").glob("*.json")):
    fx = json.loads(path.read_text())
    events = [np.asarray(e, dtype=float) for e in fx["events"]]
    D, T, decay = fx["n_nodes"], fx["end_time"], fx["decay"]
    n_jumps = fx["n_jumps"]

    model = ModelHawkesExpKernLogLik(decay=decay)
    model.fit([events], end_times=T)

    for ev in fx["evaluations"]:
        nll = negative_log_likelihood(events, ev["baseline"], ev["adjacency"], decay, T)
        predicted = ev["tick_loss"] * n_jumps + D * T
        delta = predicted - nll
        ok = abs(delta) <= 1e-9 * max(1.0, abs(nll))
        all_match &= ok
        has_excitation = any(a != 0.0 for row in ev["adjacency"] for a in row)
        print(f"{fx['name']:24s} {ev['label']:22s} {D * T:8.1f} {delta:24.16e}  "
              f"{'MATCH' if ok else 'DIFFER'}{'' if has_excitation else '  (alpha==0)'}")

print("-" * 96)
print("ALL MATCH" if all_match else "SOME DIFFER")
