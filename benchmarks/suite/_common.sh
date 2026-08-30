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
    cargo build --release --examples --manifest-path "$ROOT/Cargo.toml"
}

# bench_fit <d> <grid>
bench_fit() {
    d=$1; grid=$2
    bench_setup
    echo "=== hawk, d=$d ===" >&2
    "$ROOT/target/release/examples/bench_fit" "$WORK" "$d" "$grid"
    echo "=== tick, d=$d ===" >&2
    "$VENV/bin/python" "$ROOT/benchmarks/suite/bench_fit.py" "$WORK" "$d" "$grid"
    echo "=== scoring tick's parameters under hawk's objective ===" >&2
    "$ROOT/target/release/examples/bench_score" "$WORK" "$d"
    "$VENV/bin/python" - "$WORK" "$d" "$ROOT/benchmarks/results/fit-d$d.json" <<'PY'
import json, pathlib, sys
work, d, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
payload = {"benchmark": f"fit_d{d}",
           "methodology": "benchmarks/README.md",
           "hawk": json.loads((work / f"hawk_d{d}.json").read_text()),
           "tick": json.loads((work / f"tick_d{d}.json").read_text())}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY
}

# bench_simulate <grid> <d...>
bench_simulate() {
    grid=$1; shift
    bench_setup
    for d in "$@"; do
        echo "=== hawk simulate, d=$d ===" >&2
        "$ROOT/target/release/examples/bench_simulate" "$WORK" "$d" "$grid"
        echo "=== tick simulate, d=$d ===" >&2
        "$VENV/bin/python" "$ROOT/benchmarks/suite/bench_simulate.py" "$WORK" "$d" "$grid"
    done
    "$VENV/bin/python" - "$WORK" "$ROOT/benchmarks/results/simulate.json" "$@" <<'PY'
import json, pathlib, sys
work, out, dims = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
payload = {"benchmark": "simulate", "methodology": "benchmarks/README.md",
           "hawk": [json.loads((work / f"hawk_simulate_d{d}.json").read_text()) for d in dims],
           "tick": [json.loads((work / f"tick_simulate_d{d}.json").read_text()) for d in dims]}
out.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY
}
