"""`tick` side of the M4 simulate benchmark. See benchmarks/README.md.

Same parameters and same horizon as the hawkes side, so the expected event count
matches. The realized count differs because the two use different random number
generators; both counts are reported and the comparison is per event.

Usage: bench_simulate.py <dir> <d> <n,n,...>
"""

import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from tick.hawkes import SimuHawkesExpKernels  # noqa: E402

WARMUP = 1
TIMED = 5
SEED = 20260831
DECAY = 1.0
SINGLE_RUN_BUDGET = 600.0
CELL_BUDGET = 1800.0


def main(directory, d, grid):
    directory = pathlib.Path(directory)
    baseline = np.full(d, 0.5 / d)
    adjacency = np.full((d, d), 0.6 / d)
    rate = float(np.linalg.solve(np.eye(d) - adjacency, baseline).sum())

    runs = []
    for nominal in grid:
        horizon = nominal / rate

        def once(seed):
            sim = SimuHawkesExpKernels(adjacency=adjacency, decays=DECAY,
                                       baseline=baseline, end_time=horizon,
                                       seed=seed, verbose=False)
            start = time.perf_counter()
            sim.simulate()
            return time.perf_counter() - start, int(sum(len(t) for t in sim.timestamps))

        for repetition in range(WARMUP):
            once(SEED + repetition)

        cell_start = time.perf_counter()
        elapsed, counts, aborted = [], [], None
        for repetition in range(TIMED):
            seconds, count = once(SEED + 100 + repetition)
            counts.append(count)
            if seconds > SINGLE_RUN_BUDGET:
                aborted = "single run exceeded 600 s"
                break
            elapsed.append(seconds)
            if time.perf_counter() - cell_start > CELL_BUDGET:
                aborted = "cell exceeded 1800 s"
                break

        record = {"dimension": d, "nominal_n": nominal, "horizon": horizon}
        if aborted:
            record.update({"completed": False, "abort_reason": aborted})
            print(f"d={d} n={nominal} ABORTED: {aborted}", file=sys.stderr)
        else:
            ordered = sorted(elapsed)
            record.update({
                "completed": True,
                "seconds_median": ordered[TIMED // 2],
                "seconds_min": ordered[0],
                "seconds_max": ordered[-1],
                "seconds_all": elapsed,
                "events_median": sorted(counts)[TIMED // 2],
                "events_all": counts,
            })
            print(f"d={d} n={nominal} median={record['seconds_median']:.6f}s "
                  f"events_median={record['events_median']}", file=sys.stderr)
        runs.append(record)

    body = {"library": "tick", "benchmark": "simulate", "dimension": d,
            "seed": SEED, "warmup": WARMUP, "timed": TIMED, "runs": runs}
    (directory / f"tick_simulate_d{d}.json").write_text(json.dumps(body, indent=2) + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")])
