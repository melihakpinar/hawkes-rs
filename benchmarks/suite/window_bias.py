"""`tick` side of the OQ-5 window measurement. See benchmarks/tooling/src/bin/window_bias.rs.

Every row is the same call on the same events, because `HawkesExpKern.fit` has no
argument for the observation window. That is the finding: the column is constant by
construction, and the constant is what the caller gets no matter what was observed.

Usage: window_bias.py <dir>
"""

import json
import os
import pathlib
import sys
import warnings

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from tick.hawkes import HawkesExpKern  # noqa: E402
import inspect  # noqa: E402

TRUE_DECAY = 1.0


def main(work):
    work = pathlib.Path(work)
    hawkes = json.loads((work / "hawk_window_bias.json").read_text())
    raw = (work / "window_bias_events.txt").read_text().split()
    times = np.array([float(v) for v in raw[3:]], dtype=np.float64)

    learner = HawkesExpKern(decays=TRUE_DECAY, gofit="likelihood", penalty="l2",
                            C=1e9, solver="bfgs", tol=1e-10, max_iter=500, verbose=False)
    learner.fit([times])
    baseline = float(np.asarray(learner.baseline).ravel()[0])
    excitation = float(np.asarray(learner.adjacency).ravel()[0])

    signature = str(inspect.signature(HawkesExpKern.fit))
    rows = []
    for row in hawkes["rows"]:
        rows.append({
            "dead_time_fraction": row["dead_time_fraction"],
            "declared_horizon": row["declared_horizon"],
            "hawk_baseline": row["baseline"],
            "tick_baseline": baseline,
            "tick_excitation": excitation,
        })
        print(f"  dead_time={row['dead_time_fraction']:5.0%} "
              f"hawkes={row['baseline']:.6f}  tick={baseline:.6f}", file=sys.stderr)

    out = pathlib.Path("benchmarks/results/window-bias.json")
    out.write_text(json.dumps({
        "experiment": "window_bias",
        "question": "OQ-5",
        "methodology": "benchmarks/tooling/src/bin/window_bias.rs",
        "events": hawkes["events"],
        "observed_horizon": hawkes["observed_horizon"],
        "last_event": hawkes["last_event"],
        "true_baseline": hawkes["true_baseline"],
        "true_excitation": hawkes["true_excitation"],
        "hawkes_exp_kern_fit_signature": signature,
        "note": ("Every tick row is the same call on the same events: HawkesExpKern.fit "
                 "takes no observation window, so its estimate cannot depend on one. "
                 "The signature is recorded above so the claim is checkable."),
        "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main(sys.argv[1])
