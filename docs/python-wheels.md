# Which wheels exist, and what was actually verified

M3 steps 8, 9 and 10. This file records what a workflow was observed to build and
install, not what a matrix declaration hoped for. Anything not confirmed by a green
run is listed as unverified rather than implied.

## How the wheels are built

`abi3`, targeting CPython 3.9 as the floor (`pyo3` feature `abi3-py39`). One wheel per
platform is compiled against the 3.9 stable ABI and loaded by every later interpreter,
so the matrix is over **platforms**, not over CPython versions.

That claim is checked rather than inferred from the filename. Locally, the wheel
`hawk_hawkes-0.1.0-cp39-abi3-macosx_11_0_arm64.whl` was installed into a CPython
**3.14** environment and the full test suite passed — a five-version span on one build.
`.github/workflows/wheels.yml` repeats that on Linux across 3.9 to 3.13.

## Verified locally

| | |
| --- | --- |
| Platform | macOS 14, Apple M2, `macosx_11_0_arm64` |
| Built by | `maturin build --release` |
| Wheel | `hawk_hawkes-0.1.0-cp39-abi3-macosx_11_0_arm64.whl` |
| Installed into | a fresh venv running CPython 3.14.5 |
| Rust toolchain present | no — `PATH` reduced to `/usr/bin:/bin` |
| Source tree present | no — only `tests/` and the fixture corpus were staged |
| Result | 38 tests passed, `hawk` imported from `site-packages` |

The import location is asserted by the test run, not eyeballed: if anything resolved
`hawk` from the working directory the assertion fails.

## Verified in CI

`.github/workflows/wheels.yml` builds:

| OS | Runner | Architecture |
| --- | --- | --- |
| Linux | `ubuntu-latest` | `x86_64` (manylinux) |
| Linux | `ubuntu-24.04-arm` | `aarch64` (manylinux) |
| macOS | `macos-13` | `x86_64` |
| macOS | `macos-14` | `arm64` |
| Windows | `windows-latest` | `AMD64` |

and then installs the Linux `x86_64` wheel into CPython 3.9, 3.10, 3.11, 3.12 and 3.13,
each in an environment with no Rust toolchain and with the sources removed, and runs
`pytest` against it.

**This table is a description of the workflow, not yet of a green run.** It becomes a
statement about reality when the workflow passes; until then treat the platforms as
attempted rather than confirmed.

## Bit-for-bit results do not travel between platforms

`hawk`'s likelihood is built from `exp`, `ln` and `exp_m1`, which IEEE-754 does **not**
require to be correctly rounded and which differ in the last bits between libm
implementations. Two machines running the same wheel version can disagree by an ulp.

`+`, `-`, `*` and `/` are correctly rounded and identical everywhere, so the difference
is bounded and tiny — but it is not zero, and anything comparing results across
machines must use a tolerance. `docs/verification-log.md` records how this was found
and how FMA contraction was ruled out as the cause.

## Not covered, deliberately

- **musllinux** is skipped (`CIBW_SKIP`). Nothing has been tested against musl and
  claiming it would be inventing a platform.
- **Windows on ARM** is not built. GitHub does not offer a hosted runner for it, and
  cross-compiling without being able to run the result is exactly the kind of
  unverified claim this file exists to avoid.
- **PyPy** is not built. `pyo3`'s `abi3` wheels do not apply to it, and it has not been
  tested.
- **Publishing.** Nothing is uploaded to PyPI. The package name `hawk-hawkes` is
  reserved in `pyproject.toml` only in the sense of being written down.
