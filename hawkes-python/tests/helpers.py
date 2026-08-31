"""Shared helpers. Deliberately mirrors the Rust test helpers rather than inventing
Python-side conveniences."""

from __future__ import annotations

import math
import struct
from typing import Any

import numpy as np
import numpy.typing as npt

from hawkes import multivariate


def bits(value: float) -> int:
    """The exact IEEE-754 bit pattern, for comparisons that must not be approximate."""
    return int.from_bytes(struct.pack("<d", value), "little")


def events_of(fixture: dict[str, Any]) -> list[npt.NDArray[np.float64]]:
    return [np.asarray(component, dtype=np.float64) for component in fixture["events"]]


def parameters_of(
    fixture: dict[str, Any], evaluation: dict[str, Any]
) -> multivariate.Parameters:
    return multivariate.Parameters(
        baseline=np.asarray(evaluation["baseline"], dtype=np.float64),
        excitation=np.asarray(evaluation["adjacency"], dtype=np.float64),
        decay=float(fixture["decay"]),
    )


def computation_scale(
    fixture: dict[str, Any], evaluation: dict[str, Any]
) -> float:
    """The same denominator the Rust gate uses, transcribed.

    ``mu*T + sum_ij alpha[i][j] * sum_k (1 - exp(-beta*(T - t^j_k))) + sum |log lambda|``

    Not ``|nll|``: that is a difference of large terms and passes through zero, so a
    relative error taken against it diverges on correct code. See
    ``docs/derivations/univariate_loglikelihood.md`` §5. This is a transcription of the
    Rust helper, not a Python-side reinterpretation, and it is deliberately the *same*
    gate — a comparison that fails through the bindings and passes in Rust is a finding
    about the bindings, not a reason to loosen anything.
    """
    baseline = evaluation["baseline"]
    alpha = evaluation["adjacency"]
    beta = float(fixture["decay"])
    horizon = float(fixture["end_time"])
    events = fixture["events"]
    d = fixture["n_nodes"]

    scale = sum(baseline[i] * horizon for i in range(d))
    for i in range(d):
        for j in range(d):
            for t in events[j]:
                scale += alpha[i][j] * (1.0 - math.exp(-beta * (horizon - t)))
    for i in range(d):
        for t_k in events[i]:
            intensity = baseline[i]
            for j in range(d):
                for t_l in events[j]:
                    if t_l < t_k:
                        intensity += alpha[i][j] * beta * math.exp(-beta * (t_k - t_l))
            scale += abs(math.log(intensity))
    return scale
