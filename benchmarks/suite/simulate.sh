#!/usr/bin/env sh
# Benchmark: one simulation to a fixed horizon. See benchmarks/README.md.
# Grid fixed by §6 before any number was produced.
. "$(dirname "$0")/_common.sh"
bench_simulate 10000,100000,1000000 1 10
