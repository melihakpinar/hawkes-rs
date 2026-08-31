"""Univariate exponential-kernel Hawkes process."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hawkes import _hawkes

Parameters = _hawkes.UnivariateParameters
Fit = _hawkes.UnivariateFit

__all__ = [
    "Fit",
    "Gradient",
    "Parameters",
    "fit",
    "negative_log_likelihood",
    "negative_log_likelihood_and_gradient",
    "simulate",
]


class Gradient(NamedTuple):
    """Partial derivatives of the negative log-likelihood.

    Mirrors the Rust ``univariate::Gradient``. Units are per unit of the parameter:
    ``baseline`` is per unit intensity, ``decay`` per unit rate.
    """

    baseline: float
    excitation: float
    decay: float


def negative_log_likelihood(
    parameters: Parameters,
    times: npt.NDArray[np.float64],
    horizon: float,
) -> float:
    """Negative log-likelihood on ``[0, horizon]``.

    Args:
        parameters: the process parameters.
        times: shape ``(n,)``, dtype ``float64``, ascending, all within
            ``[0, horizon]``. Ties are permitted. The array is copied; it is never
            modified, and a non-contiguous or read-only array is accepted.
        horizon: the observation window's end, in the same time units as ``times``.
            Supplied by the caller and never inferred.

    Returns:
        The negative log-likelihood. Lower is better.

    Raises:
        TypeError: ``times`` is not a ``float64`` numpy array.
        ValueError: ``times`` is unsorted, or leaves ``[0, horizon]``, or ``horizon``
            is not positive and finite.
    """
    return _hawkes.univariate_negative_log_likelihood(parameters, times, horizon)


def negative_log_likelihood_and_gradient(
    parameters: Parameters,
    times: npt.NDArray[np.float64],
    horizon: float,
) -> tuple[float, Gradient]:
    """Negative log-likelihood and its analytic gradient, in one pass.

    Arguments are as :func:`negative_log_likelihood`.
    """
    value, baseline, excitation, decay = (
        _hawkes.univariate_negative_log_likelihood_and_gradient(parameters, times, horizon)
    )
    return value, Gradient(baseline=baseline, excitation=excitation, decay=decay)


def simulate(parameters: Parameters, horizon: float, seed: int) -> npt.NDArray[np.float64]:
    """Simulates a realization on ``[0, horizon]`` by Ogata thinning.

    Args:
        parameters: the process parameters. With a branching ratio at or above 1 the
            process is explosive and the realization can be arbitrarily large; check
            :meth:`Parameters.is_stationary` first.
        horizon: the observation window's end.
        seed: seeds a ChaCha8 generator. The same seed gives the same realization.

    Returns:
        Shape ``(n,)``, dtype ``float64``, strictly ascending. A fresh array; nothing
        aliases Rust memory.
    """
    return _hawkes.univariate_simulate(parameters, horizon, seed)


def fit(times: npt.NDArray[np.float64], horizon: float) -> Fit:
    """Maximum-likelihood fit by L-BFGS in log-parameter space.

    Stationarity is not enforced: a fitted branching ratio at or above 1 is a finding
    about the data, reported by :meth:`Fit.is_stationary`, not an error.

    On tied timestamps the objective is not a likelihood, so the maximum-likelihood
    asymptotics do not apply. The arithmetic is unaffected.

    Args:
        times: as :func:`negative_log_likelihood`.
        horizon: the observation window's end.

    Raises:
        ValueError: fewer than 3 events, or the input contract is violated.
        RuntimeError: the optimizer failed.
    """
    return _hawkes.univariate_fit(times, horizon)
