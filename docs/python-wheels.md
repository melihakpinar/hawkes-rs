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

| OS | Runner | Architecture | Wheel imported on the runner |
| --- | --- | --- | --- |
| Linux | `ubuntu-latest` | `x86_64` (manylinux) | yes |
| Linux | `ubuntu-24.04-arm` | `aarch64` (manylinux) | yes |
| macOS | `macos-14` | `arm64` | yes |
| macOS | `macos-14` | `x86_64` (cross-built, macOS 10.12+) | **no** |
| Windows | `windows-latest` | `AMD64` | yes |

and then installs the Linux `x86_64` wheel into CPython 3.9, 3.10, 3.11, 3.12 and 3.13,
each in an environment with no Rust toolchain and with the sources removed, and runs
`pytest` against it.

All five build legs and all five interpreters are green as of run
[32772566244](https://github.com/melihakpinar/hawk/actions/runs/32772566244): **38
tests passed on each interpreter**, with `hawk` asserted to resolve from
`site-packages`. One `cp39-abi3` build served 3.9 through 3.13 — the span the abi3
claim rests on, now measured rather than inferred from the wheel tag.

### What the clean-install job caught

It found two things on the first run that ever reached it, both of which mean the
job had not been testing what it said:

- **`ubuntu-latest` ships a preinstalled Rust toolchain.** The job's own guard —
  "fail if `cargo` is on `PATH`" — is what reported it. The toolchain is now removed
  before the guard runs, rather than the guard being relaxed to accommodate it. Without
  that, "installs with no Rust toolchain" would have been an untested sentence.
- **`--no-index` also cut off `numpy`.** The flag is meant to scope where
  `hawk-hawkes` may come from, since it is published nowhere and a typo in the wheel
  name should not be satisfiable from PyPI. It is not meant to block a declared runtime
  dependency, which a real user does fetch from an index.

### macOS `x86_64` is built but not executed

Its floor is macOS 10.12 rather than the 10.9 the cp39 tag defaults to, because
that is Rust's minimum for `x86_64-apple-darwin`. `delocate` refuses to ship the
mismatch, which is the right call — a 10.9 tag would promise an OS the binary
cannot load on — so the tag states the real floor.

It is cross-built on the arm64 runner and its test step is skipped, so it is a real
artifact that has never been loaded by an interpreter.

The Intel runner it used to be built on, `macos-13`, is retired. That was not obvious
from the outside: a job targeting it sat **queued for 15 hours 49 minutes without ever
starting**, while every other leg of the same run started within 6 seconds. Because
`clean-install` depends on `build`, the whole workflow could never finish, so the
verification for steps 9 and 10 was blocked by an unrelated platform's runner
availability.

This is the same standard Windows-on-ARM is excluded under, resolved the other way.
The difference is that Intel Macs are common enough that shipping an unexecuted wheel
is more useful than shipping none — provided the table says which column it falls in,
which is why that column exists.

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
