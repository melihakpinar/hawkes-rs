"""Multivariate exponential-kernel Hawkes process with cross-excitation."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hawk import _hawk

Parameters = _hawk.MultivariateParameters
Fit = _hawk.MultivariateFit

__all__ = [
    "Fit",
    "Gradient",
    "Parameters",
    "compensator_at_events",
    "fit",
    "negative_log_likelihood",
    "negative_log_likelihood_and_gradient",
    "simulate",
]

Events = list[npt.NDArray[np.float64]]


class Gradient(NamedTuple):
    """Partial derivatives of the negative log-likelihood.

    ``excitation`` has shape ``(d, d)`` and the same orientation as the parameter it
    differentiates: entry ``[i][j]`` is the derivative with respect to "j excites i".
    """

    baseline: npt.NDArray[np.float64]
    excitation: npt.NDArray[np.float64]
    decay: float


def negative_log_likelihood(
    parameters: Parameters,
    events: Events,
    horizon: float,
) -> float:
    """Negative log-likelihood on ``[0, horizon]``.

    Args:
        parameters: the process parameters.
        events: one entry per component; ``events[j]`` has shape ``(n_j,)``, dtype
            ``float64``, ascending, within ``[0, horizon]``. Ties are permitted, within
            a component and across components. Cross-component order is not defined and
            not required. A component may be empty. Arrays are copied and never
            modified.
        horizon: the observation window's end, in the same time units as the events.

    Returns:
        The negative log-likelihood. Lower is better.

    Raises:
        TypeError: a component is not a ``float64`` numpy array.
        ValueError: a component is unsorted, or leaves ``[0, horizon]``, or the
            component count does not match ``parameters.dimension``.
    """
    return _hawk.multivariate_negative_log_likelihood(parameters, events, horizon)


def negative_log_likelihood_and_gradient(
    parameters: Parameters,
    events: Events,
    horizon: float,
) -> tuple[float, Gradient]:
    """Negative log-likelihood and its analytic gradient, in one pass."""
    value, baseline, excitation, decay = (
        _hawk.multivariate_negative_log_likelihood_and_gradient(parameters, events, horizon)
    )
    return value, Gradient(baseline=baseline, excitation=excitation, decay=decay)


def compensator_at_events(
    parameters: Parameters,
    events: Events,
    horizon: float,
) -> Events:
    """The compensator evaluated at each component's own event times.

    ``result[i][k]`` is the integrated intensity of component ``i`` up to its ``k``-th
    event. For Ogata residual analysis: if the parameters are correct, the successive
    differences within each component are i.i.d. ``Exp(1)``. Test **per component** — a
    pooled test lets an error in one component be masked by another.
    """
    return _hawk.multivariate_compensator_at_events(parameters, events, horizon)


def simulate(parameters: Parameters, horizon: float, seed: int) -> Events:
    """Simulates a realization on ``[0, horizon]`` by Ogata thinning.

    Args:
        parameters: the process parameters. With a spectral radius at or above 1 the
            process is explosive; check :meth:`Parameters.is_stationary` first.
        horizon: the observation window's end.
        seed: seeds a ChaCha8 generator. The same seed gives the same realization.

    Returns:
        One array per component, each strictly ascending. Fresh arrays.
    """
    return _hawk.multivariate_simulate(parameters, horizon, seed)


def fit(events: Events, horizon: float) -> Fit:
    """Maximum-likelihood fit by L-BFGS in log-parameter space.

    The excitation matrix is optimized in log space too, so a fitted entry is never
    exactly zero: a true zero comes back as a small positive number. Exact sparsity is
    regularization, which is out of scope. See ``docs/derivations/parameter_space.md``.

    Stationarity is not enforced; the fitted spectral radius is a diagnostic, reported
    by :meth:`Fit.branching_ratio_spectral_radius`.

    Raises:
        ValueError: fewer than ``d + d*d + 1`` events in total, or the input contract
            is violated.
        RuntimeError: the optimizer failed.
    """
    return _hawk.multivariate_fit(events, horizon)
