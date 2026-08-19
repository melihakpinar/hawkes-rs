# Pinned `tick` oracle environment

`tick` is this project's differential-test oracle (CLAUDE.md §2). The reference values
in `tests/fixtures/` are produced here and nowhere else, so this image is pinned hard:
interpreter, every wheel in the transitive closure, and the CPU architecture.

## What actually worked

Determined empirically during M0, not assumed. This exact combination builds cold and
passes the smoke test:

| | |
| --- | --- |
| Base image | `python:3.13.5-slim-bookworm` |
| Platform | `linux/amd64` (**required**, see below) |
| CPython | 3.13.5 |
| `tick` | 0.8.0.2 |
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| numpydoc | 1.9.0 (**required at run time**, see below) |
| System packages | `libgomp1` only |

The complete pinned closure is `requirements.txt`, installed with
`pip install --no-deps`, so that file is the whole dependency set and not a top-level
spec.

## Three things that are not obvious

**1. numpydoc is a hard runtime dependency that `tick` does not declare.**

`tick` 0.8.0.2 lists `numpydoc` only under its `docs` extra. Without it, *every*
`tick` model class fails to construct:

```
AttributeError: 'ModelHawkesExpKernLogLik' object has no settable attribute 'dtype'
```

The cause is `tick/base/base.py:256-264`: `BaseMeta` decides which attributes are
settable by parsing class docstrings through `numpydoc.docscrape`, and bails out
returning `[]` when `docscrape is None`. Every documented attribute, `Model.dtype`
included, then goes unregistered. `pip install --no-deps numpydoc` is enough; its
Sphinx chain is not needed. Full write-up: `docs/open-questions.md` OQ-9.

**2. `linux/amd64` is required, and pinned on purpose.**

`tick` 0.8.0.2 publishes `manylinux` wheels for `x86_64` only — there is no `aarch64`
wheel. On Apple Silicon this image builds and runs under emulation. It is slow
(roughly two minutes to install) and correct. The alternative — building `tick` from
source on `arm64` — risks last-ulp differences in the reference log-likelihoods, which
is exactly what committed fixtures must not have.

**3. Likelihood fitting needs the model + solver + prox directly.**

`HawkesExpKern(gofit="likelihood")` is unusable through the documented learner
interface. `penalty="none"` installs `ProxZero`, which admits negative coefficients,
and `tick`'s C++ model then raises *"The sum of the influence on someone cannot be
negative"*. `penalty="l2"` fails the same way. Use `ModelHawkesExpKernLogLik` with a
solver and `ProxPositive`, as `smoke_test.py` does. `gofit="least-squares"` works
through the learner with `solver="agd"`.

## Build and run

```sh
docker build --platform=linux/amd64 -t hawk-tick:0.8.0.2 benchmarks/docker
docker run  --rm --platform=linux/amd64 hawk-tick:0.8.0.2
```

The smoke test simulates a seeded univariate Hawkes process on `[0, 500]`, fits it by
maximum likelihood, and prints the recovered parameters. Expected output:

```
python : 3.13.5
tick   : 0.8.0.2
numpy  : 2.5.2
simulated: [306] events on [0, 500.0]
spectral radius: 0.2
true   baseline=[0.5] adjacency=[0.2]
fitted baseline=[np.float64(0.5194768545470849)] adjacency=[np.float64(0.15158404387008273)]
loss at fitted: -0.15755019074335672
loss at truth : -0.1562764744984796
OK
```

## Regenerating the fixtures

```sh
docker run --rm --platform=linux/amd64 -v "$PWD/tests/fixtures":/out \
    hawk-tick:0.8.0.2 python /work/generate_fixtures.py --out /out
```

Output is byte-identical on re-run: every scenario fixes the simulator seed, and the
JSON is emitted with sorted keys and Python's shortest-round-trip float repr. Verify
without touching the working tree by adding `--check`, which reports `MATCH` or
`DIFFER` per file and exits non-zero on any difference.

## A caution about `loss`

`ModelHawkesExpKernLogLik.loss` is **not** the negative log-likelihood, and it is not
the formula in `tick`'s own class docstring. It is normalized by the total jump count
and carries a `-D*T` offset. See `docs/derivations/conventions.md` C7 and
`docs/open-questions.md` OQ-8 before comparing any absolute value against it.
