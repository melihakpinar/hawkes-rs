"""Every fixture that passes in Rust passes through the Python API (M3 step 2).

The gate is the same gate: the computation scale from
``docs/derivations/univariate_loglikelihood.md`` §5, not ``|nll|``, and not loosened
because this is Python. A comparison that fails here and passes in Rust is a finding
about the bindings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hawk import multivariate, univariate

from .helpers import bits, computation_scale, events_of, parameters_of

# The same constant the Rust gate uses.
TICK_TOLERANCE = 1e-9


def test_oq8_identity_holds_through_the_bindings(corpus: list[dict[str, Any]]) -> None:
    """``hawk_nll == tick_loss * n_jumps + D*T``, computed through Python.

    OQ-8. ``D`` varies across the corpus (1, 2, 3, 10), so this is testing the offset
    rather than a coincidence at one dimension.
    """
    compared = 0
    with_excitation = 0
    dimensions = set()
    for fixture in corpus:
        events = events_of(fixture)
        horizon = float(fixture["end_time"])
        d = fixture["n_nodes"]
        for evaluation in fixture["evaluations"]:
            parameters = parameters_of(fixture, evaluation)
            value = multivariate.negative_log_likelihood(parameters, events, horizon)
            from_tick = evaluation["tick_loss"] * fixture["n_jumps"] + d * horizon
            scale = computation_scale(fixture, evaluation)
            discrepancy = abs(value - from_tick)
            assert discrepancy <= TICK_TOLERANCE * scale, (
                f"{fixture['name']}/{evaluation['label']} (d={d}): hawk {value!r} vs "
                f"tick_loss*n_jumps + D*T {from_tick!r}, |difference| {discrepancy:e} "
                f"> {TICK_TOLERANCE:e} * scale {scale:e}"
            )
            compared += 1
            dimensions.add(d)
            if any(a != 0.0 for row in evaluation["adjacency"] for a in row):
                with_excitation += 1
    assert compared >= 40, f"only compared {compared} points"
    assert with_excitation >= 30
    assert dimensions >= {1, 2, 3, 10}


def test_univariate_and_multivariate_agree_bitwise_through_the_bindings(
    corpus: list[dict[str, Any]],
) -> None:
    """The ``d = 1`` equivalence, through the Python API.

    Bitwise, as in Rust. If the boundary treated a one-component multivariate call
    differently from a univariate one — a different copy path, a different array
    shape — this is where it would show.
    """
    checked = 0
    for fixture in corpus:
        if fixture["n_nodes"] != 1:
            continue
        times = np.asarray(fixture["events"][0], dtype=np.float64)
        events = events_of(fixture)
        horizon = float(fixture["end_time"])
        for evaluation in fixture["evaluations"]:
            uni = univariate.Parameters(
                float(evaluation["baseline"][0]),
                float(evaluation["adjacency"][0][0]),
                float(fixture["decay"]),
            )
            uni_value = univariate.negative_log_likelihood(uni, times, horizon)
            multi_value = multivariate.negative_log_likelihood(
                parameters_of(fixture, evaluation), events, horizon
            )
            assert bits(uni_value) == bits(multi_value), (
                f"{fixture['name']}/{evaluation['label']}: univariate {uni_value!r} vs "
                f"multivariate {multi_value!r}"
            )
            checked += 1
    assert checked >= 20, f"only checked {checked} univariate points"
