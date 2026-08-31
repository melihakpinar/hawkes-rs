"""The boundary must not change a number (M3 step 3).

This is the milestone's core. Steps 1 and 2 can both pass while the boundary silently
degrades precision — a widening cast, a copy through a lower-precision intermediate, a
reordered reduction — and only bitwise equality rules that out. Everything downstream
inherits whatever this allows.

The comparison is against ``tests/fixtures/rust-nll.json``, which records what ``hawkes``
computes in Rust as an exact bit pattern, and which ``hawkes/tests/rust_nll_manifest.rs``
fails on if it goes stale. Both sides call the same Rust function, so any difference
can only have come from the boundary.

Bit patterns rather than decimal: a decimal round trip is itself a conversion and would
hide exactly what is being looked for.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest

from hawkes import multivariate

from .helpers import bits, events_of, parameters_of


def test_manifest_covers_the_corpus(
    corpus: list[dict[str, Any]], rust_manifest: dict[tuple[str, str], int]
) -> None:
    """Guards against the comparison passing because it swept nothing."""
    expected = {
        (fixture["name"], evaluation["label"])
        for fixture in corpus
        for evaluation in fixture["evaluations"]
    }
    assert expected == set(rust_manifest), "manifest and corpus disagree about coverage"
    assert len(expected) >= 40, f"only {len(expected)} points"


def test_nll_is_bitwise_equal_to_rust(
    corpus: list[dict[str, Any]], rust_manifest: dict[tuple[str, str], int]
) -> None:
    dimensions = set()
    for fixture in corpus:
        events = events_of(fixture)
        horizon = float(fixture["end_time"])
        for evaluation in fixture["evaluations"]:
            parameters = parameters_of(fixture, evaluation)
            value = multivariate.negative_log_likelihood(parameters, events, horizon)
            expected = rust_manifest[(fixture["name"], evaluation["label"])]
            assert bits(value) == expected, (
                f"{fixture['name']}/{evaluation['label']} (d={fixture['n_nodes']}): "
                f"Python gave {value!r} (bits {bits(value):#018x}), Rust computes "
                f"{np.float64(np.frombuffer(expected.to_bytes(8, 'little'), dtype=np.float64)[0])!r} "
                f"(bits {expected:#018x}). The boundary changed a number."
            )
            dimensions.add(fixture["n_nodes"])
    assert dimensions >= {1, 2, 3, 10}, f"only compared d in {sorted(dimensions)}"


def test_gradient_is_bitwise_equal_across_two_calls(corpus: list[dict[str, Any]]) -> None:
    """The value returned by the gradient entry point must match the value-only one
    exactly, through the bindings as it does in Rust."""
    for fixture in corpus:
        events = events_of(fixture)
        horizon = float(fixture["end_time"])
        for evaluation in fixture["evaluations"]:
            parameters = parameters_of(fixture, evaluation)
            value_only = multivariate.negative_log_likelihood(parameters, events, horizon)
            with_gradient, _ = multivariate.negative_log_likelihood_and_gradient(
                parameters, events, horizon
            )
            assert bits(value_only) == bits(with_gradient), (
                f"{fixture['name']}/{evaluation['label']}: value-only {value_only!r} "
                f"vs value+gradient {with_gradient!r}"
            )


def test_a_float32_round_trip_would_be_caught(corpus: list[dict[str, Any]]) -> None:
    """The sabotage, made permanent.

    Step 3 asks for an f32 round trip to be inserted and the bitwise test confirmed
    red. Doing that by hand once proves it on the day; this asserts it on every run, so
    the bitwise test above is known to have something to detect rather than assumed to.

    The condition is data-driven, not a count. A round trip only degrades a timestamp
    that is not exactly representable in ``float32``, and four fixtures are hand-built
    from values like ``1.0`` and ``2.5`` that survive it unchanged. For those, nothing
    changing is correct. For every fixture where the timestamps *do* change, the
    likelihood must change too — otherwise the corpus has become too coarse to notice
    a precision loss and the bitwise test is no longer evidence of anything.
    """
    sensitive = 0
    for fixture in corpus:
        events = events_of(fixture)
        degraded = [c.astype(np.float32).astype(np.float64) for c in events]
        timestamps_changed = any(
            not np.array_equal(a, b) for a, b in zip(events, degraded)
        )
        horizon = float(fixture["end_time"])
        for evaluation in fixture["evaluations"]:
            parameters = parameters_of(fixture, evaluation)
            exact = multivariate.negative_log_likelihood(parameters, events, horizon)
            try:
                through_f32 = multivariate.negative_log_likelihood(
                    parameters, degraded, horizon
                )
            except ValueError:
                # A round trip can push a timestamp past the horizon or break
                # sortedness outright. That is a detection too, and a louder one.
                assert timestamps_changed
                sensitive += 1
                continue
            if not timestamps_changed:
                assert bits(exact) == bits(through_f32), (
                    f"{fixture['name']}/{evaluation['label']}: the f32 round trip left "
                    "the timestamps identical, so the likelihood must be identical too"
                )
                continue
            assert bits(exact) != bits(through_f32), (
                f"{fixture['name']}/{evaluation['label']} (n={fixture['n_jumps']}): an "
                "f32 round trip changed the timestamps but not the likelihood; the "
                "bitwise test cannot detect a precision loss on this fixture"
            )
            sensitive += 1
    assert sensitive >= 28, f"only {sensitive} points are sensitive to precision loss"
