# Benchmark tooling

The `hawkes` side of every benchmark and probe, one binary per file in `src/bin/`. This
crate is a workspace member so the binaries build with the library they measure, and it
is never published. None of these is a usage example; the example a new user runs is
`hawkes/examples/quickstart.rs`.

Build all of them with `cargo build --release -p hawkes-benchmarks`; the driver scripts
in `../suite/` do that and then run the `tick` side against the same events.

| Binary | Driven by | Produces |
| --- | --- | --- |
| `bench_fit`, `bench_score` | `suite/fit_d1.sh`, `fit_d10.sh`, `fit_d100.sh` via `_common.sh` | `results/fit-d{1,10,100}.json` |
| `bench_simulate` | `suite/simulate.sh` via `_common.sh` | `results/simulate.json` |
| `window_bias` | `suite/window_bias.sh` | `results/window-bias.json` |
| `positioning_probe`, `score_tick` | `suite/positioning_probe.sh` | `results/positioning-probe.json` |
| `inner_loop_probe` | `suite/inner_loop_probe.sh` | `results/inner-loop-probe.json` |
| `d100_diagnosis` | `suite/d100_diagnosis.sh` | `results/d100-diagnosis.json` |
| `parallel_probe` | `cargo run --release -p hawkes-benchmarks --features rayon --bin parallel_probe` | the numbers in `results/multivariate-parallel.json` |

`parallel_probe` needs the library's `rayon` feature, which this crate exposes as its
own `rayon` feature and leaves off by default so that a workspace-wide `cargo test` does
not enable it through feature unification.
