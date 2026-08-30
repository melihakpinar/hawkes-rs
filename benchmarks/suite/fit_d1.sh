#!/usr/bin/env sh
# Benchmark: one 1-component maximum-likelihood fit. See benchmarks/README.md.
# Grid fixed by §6 before any number was produced.
. "$(dirname "$0")/_common.sh"
bench_fit 1 1000,10000,100000,1000000
