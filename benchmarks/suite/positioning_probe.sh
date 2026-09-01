#!/usr/bin/env sh
# Positioning probe driver. See docs/positioning-probe.md.
set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${PROBE_WORK:-"$ROOT/target/positioning-probe"}
VENV=${PROBE_VENV:-"$ROOT/target/positioning-probe-venv"}

# §2: single-threaded on both sides. hawkes has no parallelism; this constrains tick,
# whose HawkesExpKern exposes no n_threads argument.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ ! -x "$VENV/bin/python" ]; then
    echo "creating venv at $VENV" >&2
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --disable-pip-version-check \
        'tick==0.8.0.2' 'numpydoc==1.9.0'
fi

mkdir -p "$WORK"
cargo build --release -p hawkes-benchmarks --manifest-path "$ROOT/Cargo.toml"

echo "=== hawkes ===" >&2
"$ROOT/target/release/positioning_probe" "$WORK"

echo "=== tick ===" >&2
"$VENV/bin/python" "$ROOT/benchmarks/suite/positioning_probe.py" "$WORK"

echo "=== scoring tick's parameters under hawkes's objective ===" >&2
"$ROOT/target/release/score_tick" "$WORK" > "$WORK/tick_scored.txt"
cat "$WORK/tick_scored.txt" >&2

mkdir -p "$ROOT/benchmarks/results"
"$VENV/bin/python" - "$WORK" "$ROOT/benchmarks/results/positioning-probe.json" <<'PY'
import json, pathlib, sys
work, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
hawkes = json.loads((work / "hawkes.json").read_text())
tick = json.loads((work / "tick.json").read_text())
scored = {}
for line in (work / "tick_scored.txt").read_text().split("\n"):
    if line.strip():
        n, v = line.split()
        scored[int(n)] = float(v)
for run in tick["runs"]:
    run["negative_log_likelihood_under_hawk_objective"] = scored[run["nominal_n"]]
out.write_text(json.dumps({"hawkes": hawkes, "tick": tick}, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY
