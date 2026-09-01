"""Multivariate Hawkes processes: simulation and maximum-likelihood estimation.

The API mirrors the Rust crate's and adds nothing to it. Anything that existed on one
side only would be outside the reach of the fixture corpus, which is what checks both.

Conventions, each pinned by experiment in ``docs/derivations/conventions.md``:

* The kernel is ``alpha * beta * exp(-beta * t)``, so ``alpha`` is the **branching
  ratio** itself, not ``alpha / beta``.
* ``excitation[i][j]`` means **"j excites i"**. The first index is the component being
  excited. A transposed matrix produces plausible numbers and is wrong.
* The observation window ``[0, T]`` is supplied by the caller and never inferred from
  the data. Inferring it biases the baseline upward.

Arrays must be ``numpy.float64``. That is a decision, not numpy's default, and
``docs/python-array-handling.md`` gives the reasoning: a ``float32`` timestamp array
cannot separate events near a realistic horizon and silently manufactures ties.

Exceptions raised by this package:

===================================  ==========================================
Exception                            Raised when
===================================  ==========================================
``TypeError``                        an array is not ``float64``, or is not an array
``ValueError``                       the input contract is violated: non-positive or
                                     non-finite parameters, unsorted timestamps, a
                                     timestamp outside ``[0, horizon]``, a
                                     non-positive horizon, a dimension mismatch, a
                                     negative excitation entry, or too few events to
                                     identify the parameters
``RuntimeError``                     the optimizer failed as a solver
===================================  ==========================================

No Rust panic reaches the interpreter; ``hawkes``'s library code returns ``Result``
rather than panicking (CLAUDE.md §5), and the bindings map every variant explicitly.
"""

from importlib.metadata import version

from hawkes import multivariate, univariate

__all__ = ["multivariate", "univariate"]
# The distribution's own metadata, so the number is written once, in pyproject.toml.
__version__ = version("hawkes-rs")
