# hawkes — Python bindings

Multivariate Hawkes processes: simulation and maximum-likelihood estimation.

> **Pre-alpha.** The API will change. Do not depend on this package.

```python
import numpy as np
from hawkes import multivariate

parameters = multivariate.Parameters(
    baseline=np.array([0.3, 0.2]),
    # excitation[i][j] is "j excites i"
    excitation=np.array([[0.10, 0.45], [0.05, 0.10]]),
    decay=1.4,
)
events = multivariate.simulate(parameters, horizon=5000.0, seed=1)
fitted = multivariate.fit(events, horizon=5000.0)

print(fitted.parameters.excitation)
print(fitted.branching_ratio_spectral_radius(), fitted.converged)
```

## Conventions that will bite you if you assume the other one

Each is pinned by experiment in the repository's `docs/derivations/conventions.md`.

- The kernel is `alpha * beta * exp(-beta * t)`, so **`alpha` is the branching ratio
  itself**, not `alpha / beta`. Some of the literature uses the other normalization;
  [Laub2015] does.
- **`excitation[i][j]` means "j excites i".** The first index is the component being
  excited. A transposed matrix produces plausible numbers and is wrong, and is only
  detectable on asymmetric data.
- The observation window `[0, T]` is **supplied by you and never inferred**. Taking
  `T = max(events)` discards the trailing dead time and biases the baseline upward.

## Arrays

`float64` only, and that is a decision rather than numpy's default: a `float32`
timestamp array cannot separate events near a realistic horizon and silently
manufactures ties, which invalidates the maximum-likelihood asymptotics. Convert
explicitly with `np.asarray(x, dtype=np.float64)`.

Non-contiguous, Fortran-ordered and read-only arrays are all accepted and copied.
Logical indexing is always honoured, so an F-ordered excitation matrix is *not*
transposed. The reasoning is in `docs/python-array-handling.md`.

## Licence

MIT or Apache-2.0, at your option.
