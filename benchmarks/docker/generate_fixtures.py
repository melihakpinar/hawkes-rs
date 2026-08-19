"""Emit reference fixtures from the pinned `tick` oracle.

Each fixture records a simulated Hawkes realization together with the values `tick`
reports for it. Rust's differential test (hawk/tests/differential_tick.rs) replays
these and must agree.

Determinism: every scenario fixes the simulator seed, and the JSON is written with
sorted keys and Python's shortest-round-trip float repr, so re-running this script
reproduces byte-identical files. That is asserted by --check.

Conventions recorded here are pinned to tick's source, not assumed. See
docs/derivations/conventions.md.

Run with:
    docker run --rm --platform=linux/amd64 -v "$PWD/tests/fixtures":/out \
        hawk-tick:0.8.0.2 python /work/generate_fixtures.py --out /out
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from tick.hawkes import ModelHawkesExpKernLogLik, SimuHawkesExpKernels

import tick

SCHEMA_VERSION = 1

# Scenarios span univariate and multivariate, small and large event counts, and
# symmetric and asymmetric excitation. The asymmetric ones are load-bearing: a
# transposed adjacency matrix produces plausible-looking numbers and is only caught
# when alpha[i][j] != alpha[j][i].
SCENARIOS = [
    {
        "name": "univariate_tiny",
        "description": "Univariate, ~10 events. Small enough to check by hand.",
        "baseline": [0.4],
        "adjacency": [[0.3]],
        "decay": 1.0,
        "end_time": 20.0,
        "seed": 20240001,
    },
    {
        "name": "univariate_small",
        "description": "Univariate, moderate branching ratio 0.2.",
        "baseline": [0.5],
        "adjacency": [[0.2]],
        "decay": 1.5,
        "end_time": 200.0,
        "seed": 20240002,
    },
    {
        "name": "univariate_large",
        "description": "Univariate, branching ratio 0.5, long horizon.",
        "baseline": [0.8],
        "adjacency": [[0.5]],
        "decay": 2.0,
        "end_time": 2000.0,
        "seed": 20240003,
    },
    {
        "name": "bivariate_symmetric",
        "description": "Two nodes, symmetric adjacency.",
        "baseline": [0.3, 0.3],
        "adjacency": [[0.2, 0.1], [0.1, 0.2]],
        "decay": 1.0,
        "end_time": 300.0,
        "seed": 20240004,
    },
    {
        "name": "bivariate_asymmetric",
        "description": (
            "Two nodes, strongly asymmetric: node 1 excites node 0 much more than "
            "the reverse. Detects a transposed adjacency matrix."
        ),
        "baseline": [0.2, 0.5],
        "adjacency": [[0.1, 0.6], [0.05, 0.15]],
        "decay": 1.2,
        "end_time": 400.0,
        "seed": 20240005,
    },
    {
        "name": "trivariate_asymmetric",
        "description": "Three nodes, asymmetric, cyclic excitation.",
        "baseline": [0.2, 0.3, 0.25],
        "adjacency": [[0.05, 0.3, 0.0], [0.0, 0.05, 0.35], [0.25, 0.0, 0.05]],
        "decay": 1.8,
        "end_time": 500.0,
        "seed": 20240006,
    },
]

# Parameter points at which the oracle values are recorded. Recording the loss at
# perturbed parameters as well as at the truth pins the *shape* of the likelihood
# surface, not just one number: an implementation with a compensating error at the
# true parameters still has to match everywhere else.
PERTURBATIONS = [
    ("truth", 1.0, 1.0),
    ("baseline_scaled_up", 1.7, 1.0),
    ("adjacency_scaled_down", 1.0, 0.4),
    ("both_scaled", 0.6, 1.5),
]


def to_coeffs(baseline, adjacency):
    """Flatten to tick's coefficient vector.

    Layout is `[baseline (D), adjacency.ravel() (D*D)]` in C order, per
    tick/hawkes/inference/base/learner_hawkes_param.py:227 (`coeffs[:n_nodes]`) and
    tick/hawkes/inference/hawkes_expkern_fixeddecay.py:197-198
    (`coeffs[n_nodes:].reshape((n_nodes, n_nodes))`).
    """
    return np.concatenate([np.asarray(baseline, float),
                           np.asarray(adjacency, float).ravel()])


def build(scenario):
    baseline = np.array(scenario["baseline"], dtype=float)
    adjacency = np.array(scenario["adjacency"], dtype=float)
    decay = float(scenario["decay"])
    end_time = float(scenario["end_time"])
    n_nodes = baseline.size

    simulator = SimuHawkesExpKernels(
        adjacency=adjacency,
        decays=np.full_like(adjacency, decay),
        baseline=baseline,
        end_time=end_time,
        seed=scenario["seed"],
        verbose=False,
    )
    simulator.simulate()
    events = [np.asarray(t, dtype=float) for t in simulator.timestamps]

    model = ModelHawkesExpKernLogLik(decay=decay)
    # end_times is passed explicitly. Left as None, tick infers the window as
    # max(events) (tick/hawkes/model/base/model_hawkes.py:88-91), discarding the
    # trailing dead time and biasing the baseline upward.
    model.fit([events], end_times=end_time)

    evaluations = []
    for label, baseline_factor, adjacency_factor in PERTURBATIONS:
        point_baseline = baseline * baseline_factor
        point_adjacency = adjacency * adjacency_factor
        coeffs = to_coeffs(point_baseline, point_adjacency)
        gradient = model.grad(coeffs, out=np.zeros(coeffs.size))
        evaluations.append({
            "label": label,
            "baseline": [float(v) for v in point_baseline],
            "adjacency": [[float(v) for v in row] for row in point_adjacency],
            "coeffs": [float(v) for v in coeffs],
            "tick_loss": float(model.loss(coeffs)),
            "tick_grad": [float(v) for v in gradient],
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "name": scenario["name"],
        "description": scenario["description"],
        "generator": {
            "script": "benchmarks/docker/generate_fixtures.py",
            "tick_version": tick.__version__,
            "numpy_version": np.__version__,
            "python_version": ".".join(str(v) for v in sys.version_info[:3]),
            "seed": scenario["seed"],
        },
        # tick's ModelHawkesExpKernLogLik.loss is NOT the plain negative
        # log-likelihood. Measured against the closed-form Poisson case
        # (adjacency == 0) it is
        #     loss = (1/n_jumps) * sum_i [ int_0^T (lambda_i(t) - 1) dt
        #                                  - sum_k log lambda_i(t_k) ],
        # i.e. the negative log-likelihood *ratio against a unit-rate Poisson
        # process*, normalized by the total jump count. That differs from the
        # formula in tick's own docstring by exactly D*T/n_jumps. See
        # docs/open-questions.md OQ-8: the offset is confirmed for adjacency == 0
        # and must be re-confirmed for adjacency != 0 in M1.
        "tick_loss_convention": (
            "negative log-likelihood ratio w.r.t. unit-rate Poisson, "
            "divided by n_jumps; see docs/open-questions.md OQ-8"
        ),
        "n_nodes": int(n_nodes),
        "decay": decay,
        "end_time": end_time,
        "baseline": [float(v) for v in baseline],
        "adjacency": [[float(v) for v in row] for row in adjacency],
        "spectral_radius": float(simulator.spectral_radius()),
        "n_jumps": int(sum(len(t) for t in events)),
        "events": [[float(v) for v in t] for t in events],
        "evaluations": evaluations,
    }


def render(fixture):
    return json.dumps(fixture, indent=2, sort_keys=True) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--check", action="store_true",
                        help="fail if any file on disk differs from what would "
                             "be written")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for scenario in SCENARIOS:
        fixture = build(scenario)
        text = render(fixture)
        path = args.out / f"{scenario['name']}.json"
        if args.check:
            existing = path.read_text() if path.exists() else None
            status = "MATCH" if existing == text else "DIFFER"
            failures += status == "DIFFER"
            print(f"{status} {path.name} ({fixture['n_jumps']} events)")
        else:
            path.write_text(text)
            print(f"wrote {path.name} "
                  f"({fixture['n_jumps']} events, {fixture['n_nodes']} nodes)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
