#!/usr/bin/env sh
# Shared driver for the M4 benchmarks. See benchmarks/README.md.
# Sourced by fit_d1.sh, fit_d10.sh, fit_d100.sh and simulate.sh.
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${BENCH_WORK:-"$ROOT/target/bench-work"}
VENV=${BENCH_VENV:-"$ROOT/target/bench-venv"}

# benchmarks/README.md §2: single-threaded on both sides.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

bench_setup() {
    if [ ! -x "$VENV/bin/python" ]; then
        echo "creating venv at $VENV" >&2
        python3 -m venv "$VENV"
        "$VENV/bin/pip" install --quiet --disable-pip-version-check \
            'tick==0.8.0.2' 'numpydoc==1.9.0'
    fi
    mkdir -p "$WORK" "$ROOT/benchmarks/results"
    cargo build --release -p hawkes-benchmarks --manifest-path "$ROOT/Cargo.toml"
}

# §4.1's cell budget, enforced by killing the process. A budget that cannot interrupt
# a call in progress is not a budget: tick's d=100 least-squares fit was observed
# running past 40 minutes inside a single fit() call, where no in-process check can
# fire. Each grid point therefore runs as its own process and is killed at the cap.
CELL_BUDGET=${BENCH_CELL_BUDGET:-1800}

run_capped() {
    "$@" &
    pid=$!
    ( sleep "$CELL_BUDGET"; kill -9 "$pid" 2>/dev/null ) 2>/dev/null &
    watcher=$!
    wait "$pid" 2>/dev/null; status=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null || true
    if [ "$status" -ne 0 ]; then
        echo "  cell exceeded ${CELL_BUDGET}s or failed (status $status); recorded as not completed" >&2
    fi
    return 0
}

# bench_fit <d> <grid>
bench_fit() {
    d=$1; grid=$2
    bench_setup
    for n in $(echo "$grid" | tr ',' ' '); do
        echo "=== d=$d n=$n ===" >&2
        run_capped "$ROOT/target/release/bench_fit" "$WORK" "$d" "$n"
        run_capped "$VENV/bin/python" "$ROOT/benchmarks/suite/bench_fit.py" "$WORK" "$d" "$n"
        run_capped "$ROOT/target/release/bench_score" "$WORK" "$d" "$n"
    done
    "$VENV/bin/python" - "$WORK" "$d" "$grid" "$ROOT/benchmarks/results/fit-d$d.json" <<'PY'
import json, pathlib, sys
work, d, grid, out = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4])
hawkes, tick = [], []
for n in grid.split(","):
    hp, tp = work / f"hawk_d{d}_n{n}.json", work / f"tick_d{d}_n{n}.json"
    if hp.exists():
        hawkes.append(json.loads(hp.read_text()))
    else:
        hawkes.append({"library": "hawkes", "dimension": int(d),
                     "run": {"dimension": int(d), "nominal_n": int(n), "completed": False,
                             "abort_reason": "process killed at the cell budget"}})
    if tp.exists():
        tick.append(json.loads(tp.read_text()))
    else:
        tick.append({"library": "tick", "dimension": int(d),
                     "runs": [{"dimension": int(d), "nominal_n": int(n), "completed": False,
                               "abort_reason": "process killed at the cell budget"}]})
out.write_text(json.dumps({"benchmark": f"fit_d{d}",
                           "methodology": "benchmarks/README.md",
                           "cell_budget_seconds": 1800,
                           "hawkes": hawkes, "tick": tick}, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY
}

# bench_simulate <grid> <d...>
bench_simulate() {
    grid=$1; shift
    bench_setup
    for d in "$@"; do
        echo "=== hawkes simulate, d=$d ===" >&2
        "$ROOT/target/release/bench_simulate" "$WORK" "$d" "$grid"
        echo "=== tick simulate, d=$d ===" >&2
        "$VENV/bin/python" "$ROOT/benchmarks/suite/bench_simulate.py" "$WORK" "$d" "$grid"
    done
    "$VENV/bin/python" - "$WORK" "$ROOT/benchmarks/results/simulate.json" "$@" <<'PY'
import json, pathlib, sys
work, out, dims = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
payload = {"benchmark": "simulate", "methodology": "benchmarks/README.md",
           "hawkes": [json.loads((work / f"hawk_simulate_d{d}.json").read_text()) for d in dims],
           "tick": [json.loads((work / f"tick_simulate_d{d}.json").read_text()) for d in dims]}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY
}
