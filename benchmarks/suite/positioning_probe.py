"""`tick` side of the positioning probe. See docs/positioning-probe.md.

Reads the events written by the hawk side, so both fit exactly the same data.
Times HawkesExpKern only; loading and construction are outside the clock.
"""

import json
import os
import pathlib
import platform
import sys
import time

import numpy as np
import scipy
import tick
from tick.hawkes import HawkesExpKern

# Fixed by docs/positioning-probe.md §4 and §5.
WARMUP = 1
TIMED = 5
TRUE_DECAY = 1.0          # §5.1: tick is handed the true beta, it cannot fit it
PENALTY_C = 1e9           # §5.2: agrees with C=1e12 to nine significant figures
TOL = 1e-10               # §5.3
MAX_ITER = 500


def main(directory):
    directory = pathlib.Path(directory)
    runs = []
    params_out = []

    for nominal in (1_000, 10_000, 100_000, 1_000_000):
        raw = (directory / f"events_{nominal}.txt").read_text().split()
        horizon = float(raw[0])
        times = np.array([float(v) for v in raw[1:]], dtype=float)
        events = [times]

        def make():
            return HawkesExpKern(
                decays=TRUE_DECAY,
                gofit="likelihood",
                penalty="l2",
                C=PENALTY_C,
                solver="bfgs",
                tol=TOL,
                max_iter=MAX_ITER,
                verbose=False,
            )

        for _ in range(WARMUP):
            make().fit(events)

        elapsed = []
        learner = None
        for _ in range(TIMED):
            learner = make()
            start = time.perf_counter()
            learner.fit(events)
            elapsed.append(time.perf_counter() - start)
        elapsed.sort()

        baseline = float(learner.baseline[0])
        adjacency = float(learner.adjacency.ravel()[0])
        median = elapsed[TIMED // 2]
        print(
            f"n={nominal:<9} events={len(times):<9} median={median:.6f}s  "
            f"mu={baseline:.10f} alpha={adjacency:.10f} beta={TRUE_DECAY:.10f} (fixed)",
            file=sys.stderr,
        )

        runs.append({
            "nominal_n": nominal,
            "events": len(times),
            "horizon": horizon,
            "seconds_median": median,
            "seconds_min": elapsed[0],
            "seconds_max": elapsed[-1],
            "seconds_all": elapsed,
            "baseline": baseline,
            "excitation": adjacency,
            "decay": TRUE_DECAY,
            "decay_was_fitted": False,
        })
        params_out.append(f"{nominal} {baseline!r} {adjacency!r} {TRUE_DECAY!r}")

    (directory / "tick_params.txt").write_text("\n".join(params_out) + "\n")
    (directory / "tick.json").write_text(json.dumps({
        "side": "tick",
        "warmup": WARMUP,
        "timed": TIMED,
        "tick_version": tick.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "settings": {
            "gofit": "likelihood", "penalty": "l2", "C": PENALTY_C,
            "solver": "bfgs", "tol": TOL, "max_iter": MAX_ITER,
            "decays": TRUE_DECAY, "decay_is_fixed_input": True,
        },
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "runs": runs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
