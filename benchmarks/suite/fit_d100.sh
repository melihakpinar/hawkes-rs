#!/usr/bin/env sh
# Benchmark: one 100-component maximum-likelihood fit. See benchmarks/README.md.
# Grid fixed by §6 before any number was produced.
. "$(dirname "$0")/_common.sh"
bench_fit 100 100000
