# Benchmark suite

One runnable script per benchmark (CLAUDE.md §7). Each writes a JSON result to
`benchmarks/results/`, which is committed so that regressions are visible in diffs.

Empty as of M0: there is nothing to benchmark yet. `hawkes` ships no algorithms until
M1, and speed is explicitly secondary to correctness.

Scripts that produce the `tick` **reference fixtures** are not benchmarks and live in
`benchmarks/docker/` with the oracle image, since they must run inside it.
