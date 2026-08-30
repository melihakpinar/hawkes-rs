#!/usr/bin/env sh
# OQ-5: can each library express an observation window? See benchmarks/README.md.
. "$(dirname "$0")/_common.sh"
bench_setup
"$ROOT/target/release/examples/window_bias" "$WORK"
"$VENV/bin/python" "$ROOT/benchmarks/suite/window_bias.py" "$WORK"
