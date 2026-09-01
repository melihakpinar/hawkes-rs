#!/usr/bin/env sh
# The two d = 100 questions. See the doc comment of
# benchmarks/tooling/src/bin/d100_diagnosis.rs; writes benchmarks/results/d100-diagnosis.json.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cargo build --release -p hawkes-benchmarks --manifest-path "$ROOT/Cargo.toml"
"$ROOT/target/release/d100_diagnosis" "$ROOT/benchmarks/results"
