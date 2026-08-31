#!/usr/bin/env sh
# Positioning probe part 2. See docs/positioning-probe.md.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${PROBE_WORK:-"$ROOT/target/positioning-probe"}
VENV=${PROBE_VENV:-"$ROOT/target/positioning-probe-venv"}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --disable-pip-version-check \
        'tick==0.8.0.2' 'numpydoc==1.9.0'
fi

mkdir -p "$WORK"
cargo build --release --examples --manifest-path "$ROOT/Cargo.toml"

echo "=== hawkes ===" >&2
"$ROOT/target/release/examples/inner_loop_probe" "$WORK"
echo "=== tick ===" >&2
"$VENV/bin/python" "$ROOT/benchmarks/suite/inner_loop_probe.py" "$WORK"

mkdir -p "$ROOT/benchmarks/results"
"$VENV/bin/python" - "$WORK" "$ROOT/benchmarks/results/inner-loop-probe.json" <<'PY'
import json, pathlib, sys
work, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.write_text(json.dumps({
    "hawkes": json.loads((work / "hawk_inner.json").read_text()),
    "tick": json.loads((work / "tick_inner.json").read_text()),
}, indent=2) + "\n")
print(f"wrote {out}", file=sys.stderr)
PY
