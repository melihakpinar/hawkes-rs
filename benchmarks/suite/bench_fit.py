"""`tick` side of the M4 fit benchmarks. See benchmarks/README.md.

Reads the events written by the hawk side, so both fit exactly the same data.
Times the fit call only; loading and construction are outside the clock.

Usage: bench_fit.py <dir> <d> <n,n,...>
"""

import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np

# benchmarks/README.md §2: single-threaded on both sides.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from tick.hawkes import HawkesExpKern  # noqa: E402

# §4 and §5, fixed before any number was produced.
WARMUP = 1
TIMED = 5
TRUE_DECAY = 1.0        # §5.1: tick is handed the true beta; it cannot fit it
PENALTY_C = 1e9         # §5.2
TOL = 1e-10             # §5.3
MAX_ITER = 500
SINGLE_RUN_BUDGET = 600.0    # §4.1
CELL_BUDGET = 1800.0         # §4.1


def read_events(path):
    values = path.read_text().split()
    horizon = float(values[0])
    d = int(values[1])
    events, cursor = [], 2
    for _ in range(d):
        count = int(values[cursor]); cursor += 1
        events.append(np.array([float(v) for v in values[cursor:cursor + count]],
                               dtype=np.float64))
        cursor += count
    return horizon, events


def run_one(events, gofit):
    """Warmup then TIMED fits, honouring the §4.1 budget. Returns a record."""
    def make():
        return HawkesExpKern(decays=TRUE_DECAY, gofit=gofit, penalty="l2",
                             C=PENALTY_C, solver="bfgs", tol=TOL,
                             max_iter=MAX_ITER, verbose=False)

    cell_start = time.perf_counter()
    try:
        for _ in range(WARMUP):
            make().fit(events)
    except Exception as exc:                       # §5.4: record, do not retry
        return {"completed": False, "gofit": gofit,
                "abort_reason": f"{type(exc).__name__}: {exc}".strip()}

    elapsed, learner = [], None
    while len(elapsed) < TIMED:
        learner = make()
        start = time.perf_counter()
        try:
            learner.fit(events)
        except Exception as exc:
            return {"completed": False, "gofit": gofit,
                    "abort_reason": f"{type(exc).__name__}: {exc}".strip()}
        seconds = time.perf_counter() - start
        if seconds > SINGLE_RUN_BUDGET:
            return {"completed": False, "gofit": gofit,
                    "abort_reason": "single run exceeded 600 s"}
        elapsed.append(seconds)
        if time.perf_counter() - cell_start > CELL_BUDGET:
            return {"completed": False, "gofit": gofit,
                    "abort_reason": "cell exceeded 1800 s"}

    ordered = sorted(elapsed)
    return {
        "completed": True,
        "gofit": gofit,
        "seconds_median": ordered[TIMED // 2],
        "seconds_min": ordered[0],
        "seconds_max": ordered[-1],
        "seconds_all": elapsed,
        "baseline": [float(v) for v in np.asarray(learner.baseline).ravel()],
        "excitation": [float(v) for v in np.asarray(learner.adjacency).ravel()],
        "decay": TRUE_DECAY,
    }


def main(directory, d, grid):
    directory = pathlib.Path(directory)
    # §5.4: the likelihood objective is only reachable at d = 1. Least-squares is
    # tick's default and the only estimator that runs at d > 1. Both are recorded at
    # d = 1 so the cost of the objective difference is visible where both work.
    objectives = ["likelihood", "least-squares"] if d == 1 else ["least-squares"]

    for nominal in grid:
        horizon, events = read_events(directory / f"events_d{d}_n{nominal}.txt")
        realized = int(sum(len(c) for c in events))
        runs = []
        for gofit in objectives:
            record = {"dimension": d, "nominal_n": nominal, "events": realized,
                      "horizon": horizon}
            record.update(run_one(events, gofit))
            runs.append(record)
            if record["completed"]:
                print(f"d={d} n={nominal} events={realized} gofit={gofit} "
                      f"median={record['seconds_median']:.6f}s", file=sys.stderr)
            else:
                print(f"d={d} n={nominal} gofit={gofit} ABORTED: "
                      f"{record['abort_reason'][:90]}", file=sys.stderr)
        # One file per grid point, written as soon as that point finishes, so a cell
        # killed at the §4.1 budget loses only the cell in flight (see _common.sh).
        body = {"library": "tick", "dimension": d, "warmup": WARMUP, "timed": TIMED,
                "penalty_C": PENALTY_C, "tol": TOL, "max_iter": MAX_ITER,
                "decay_is_input": True, "runs": runs}
        (directory / f"tick_d{d}_n{nominal}.json").write_text(
            json.dumps(body, indent=2) + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")])
