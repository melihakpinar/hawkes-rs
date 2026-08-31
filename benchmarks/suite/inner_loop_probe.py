"""`tick` side of positioning probe part 2. See docs/positioning-probe.md part 2.

Times one loss_and_grad() on ModelHawkesExpKernLogLik, on the same events the hawkes
side wrote out, and probes the weight-cache behaviour by timing.
"""

import json
import os
import pathlib
import platform
import statistics
import sys
import time

import numpy as np
import tick
from tick.hawkes import ModelHawkesExpKernLogLik

WARMUP = 1
TIMED = 5
TRUE_BASELINE = 0.5
TRUE_EXCITATION = 0.6
TRUE_DECAY = 1.0
CACHE_PROBE_CALLS = 10


def median_of(fn, warmup=WARMUP, timed=TIMED):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(timed):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples


def main(directory):
    directory = pathlib.Path(directory)
    runs = []

    for nominal in (10_000, 100_000, 1_000_000):
        raw = (directory / f"events_{nominal}.txt").read_text().split()
        horizon = float(raw[0])
        times = np.array([float(v) for v in raw[1:]], dtype=float)
        events = [times]
        coeffs = np.array([TRUE_BASELINE, TRUE_EXCITATION])

        # Steady state: a model whose weights are already computed. This is the call
        # the optimizer makes on every iteration after the first.
        warm = ModelHawkesExpKernLogLik(decay=TRUE_DECAY, n_threads=1)
        warm.fit(events, end_times=horizon)
        warm.loss_and_grad(coeffs, out=np.zeros(2))  # force compute_weights

        lg = median_of(lambda: warm.loss_and_grad(coeffs, out=np.zeros(2)))
        loss_only = median_of(lambda: warm.loss(coeffs))
        grad_only = median_of(lambda: warm.grad(coeffs, out=np.zeros(2)))

        # --- cache probe -------------------------------------------------------
        # (1) cost of set_data itself
        set_data = []
        for _ in range(TIMED):
            m = ModelHawkesExpKernLogLik(decay=TRUE_DECAY, n_threads=1)
            start = time.perf_counter()
            m.fit(events, end_times=horizon)
            set_data.append(time.perf_counter() - start)
        set_data.sort()

        # (2) first loss() on a fresh model vs the calls that follow it
        first_calls, later_calls = [], []
        for _ in range(TIMED):
            m = ModelHawkesExpKernLogLik(decay=TRUE_DECAY, n_threads=1)
            m.fit(events, end_times=horizon)
            start = time.perf_counter()
            m.loss(coeffs)
            first_calls.append(time.perf_counter() - start)
            seq = []
            for _ in range(CACHE_PROBE_CALLS):
                start = time.perf_counter()
                m.loss(coeffs)
                seq.append(time.perf_counter() - start)
            later_calls.append(statistics.median(seq))
        first_calls.sort()
        later_calls.sort()

        # (3) first loss() after mutating decay, vs steady state on the same object
        after_decay_change, steady_same_object = [], []
        m = ModelHawkesExpKernLogLik(decay=TRUE_DECAY, n_threads=1)
        m.fit(events, end_times=horizon)
        m.loss(coeffs)
        for i in range(TIMED):
            m.decay = TRUE_DECAY + 0.001 * (i + 1)
            start = time.perf_counter()
            m.loss(coeffs)
            after_decay_change.append(time.perf_counter() - start)
            seq = []
            for _ in range(CACHE_PROBE_CALLS):
                start = time.perf_counter()
                m.loss(coeffs)
                seq.append(time.perf_counter() - start)
            steady_same_object.append(statistics.median(seq))
        after_decay_change.sort()
        steady_same_object.sort()

        mid = TIMED // 2
        print(
            f"n={nominal:<9} events={len(times):<9} "
            f"loss_and_grad={lg[mid]:.9f}s  loss={loss_only[mid]:.9f}  "
            f"grad={grad_only[mid]:.9f}",
            file=sys.stderr,
        )
        print(
            f"          cache probe: set_data={set_data[mid]:.9f}  "
            f"first_loss={first_calls[mid]:.9f}  later_loss={later_calls[mid]:.9f}  "
            f"after_decay_change={after_decay_change[mid]:.9f}  "
            f"steady={steady_same_object[mid]:.9f}",
            file=sys.stderr,
        )

        runs.append({
            "nominal_n": nominal,
            "events": len(times),
            "loss_and_grad_seconds_median": lg[mid],
            "loss_and_grad_seconds_min": lg[0],
            "loss_and_grad_seconds_max": lg[-1],
            "loss_and_grad_seconds_all": lg,
            "loss_only_seconds_median": loss_only[mid],
            "grad_only_seconds_median": grad_only[mid],
            "cache_probe": {
                "set_data_seconds_median": set_data[mid],
                "first_loss_after_set_data_median": first_calls[mid],
                "subsequent_loss_median": later_calls[mid],
                "first_loss_after_decay_change_median": after_decay_change[mid],
                "steady_loss_same_object_median": steady_same_object[mid],
            },
        })

    (directory / "tick_inner.json").write_text(json.dumps({
        "side": "tick",
        "warmup": WARMUP,
        "timed": TIMED,
        "cache_probe_calls": CACHE_PROBE_CALLS,
        "tick_version": tick.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "n_threads": 1,
        "evaluated_at": {"baseline": TRUE_BASELINE, "excitation": TRUE_EXCITATION,
                         "decay": TRUE_DECAY},
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "runs": runs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
