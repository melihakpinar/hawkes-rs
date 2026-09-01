"""The input contract, enforced at the boundary (M3 steps 4 and 6).

Each test names the rule from ``docs/derivations/conventions.md`` C8 that it enforces.
The contract is the Rust one; the boundary must not widen or narrow it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from hawkes import multivariate, univariate

from .helpers import bits


def _uni() -> univariate.Parameters:
    return univariate.Parameters(0.5, 0.4, 1.0)


def _multi(d: int = 2) -> multivariate.Parameters:
    return multivariate.Parameters(
        baseline=np.full(d, 0.4),
        excitation=np.full((d, d), 0.1),
        decay=1.0,
    )


# --- C8 rule 1: timestamps ascending within a component; unsorted is an error -------


def test_unsorted_timestamps_are_rejected_not_sorted() -> None:
    """C8 rule 1. `hawkes` rejects rather than silently sorting.

    `tick` accepts unsorted input and returns a different wrong number for each
    ordering, with no error (OQ-7). The boundary must not reintroduce that.
    """
    with pytest.raises(ValueError, match="ascending"):
        univariate.negative_log_likelihood(_uni(), np.array([2.0, 1.0]), 5.0)
    with pytest.raises(ValueError, match="ascending"):
        multivariate.negative_log_likelihood(
            _multi(), [np.array([1.0, 2.0]), np.array([3.0, 2.5])], 5.0
        )


def test_a_sorted_array_and_its_shuffle_do_not_both_succeed() -> None:
    """The failure mode C8 exists to prevent, stated directly."""
    ordered = np.array([1.0, 2.0, 2.5])
    assert univariate.negative_log_likelihood(_uni(), ordered, 5.0) == pytest.approx(
        univariate.negative_log_likelihood(_uni(), ordered.copy(), 5.0)
    )
    for shuffled in (np.array([2.5, 2.0, 1.0]), np.array([2.0, 1.0, 2.5])):
        with pytest.raises(ValueError):
            univariate.negative_log_likelihood(_uni(), shuffled, 5.0)


# --- C8 rule 4: every timestamp in [0, T], endpoints included -----------------------


def test_a_timestamp_past_the_horizon_is_rejected() -> None:
    """C8 rule 4."""
    with pytest.raises(ValueError, match="outside the window"):
        univariate.negative_log_likelihood(_uni(), np.array([1.0, 6.0]), 5.0)


def test_a_negative_timestamp_is_rejected() -> None:
    """C8 rule 4."""
    with pytest.raises(ValueError, match="outside the window"):
        univariate.negative_log_likelihood(_uni(), np.array([-1.0, 1.0]), 5.0)


def test_the_window_endpoints_are_accepted() -> None:
    """C8 rule 4: endpoints included. An event at exactly `T` contributes
    `1 - exp(0) = 0` to the compensator, which is a value, not an error."""
    value = univariate.negative_log_likelihood(_uni(), np.array([0.0, 2.5, 5.0]), 5.0)
    assert np.isfinite(value)


def test_a_non_positive_horizon_is_rejected() -> None:
    """C5: the window is supplied by the caller and must be a window."""
    for horizon in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="horizon"):
            univariate.negative_log_likelihood(_uni(), np.array([1.0]), horizon)


# --- C8 rule 3: ties accepted, within a component and across components -------------


def test_ties_within_a_component_are_accepted() -> None:
    """C8 rule 3. Simultaneous events do not excite each other (C3); that is the
    arithmetic, and it is tested in Rust. Here: they are not an error."""
    value = univariate.negative_log_likelihood(_uni(), np.array([1.0, 2.0, 2.0, 3.0]), 5.0)
    assert np.isfinite(value)


def test_ties_across_components_are_accepted() -> None:
    """C8 rule 3, the cross-component case."""
    value = multivariate.negative_log_likelihood(
        _multi(), [np.array([1.0, 2.5]), np.array([2.5, 4.0])], 6.0
    )
    assert np.isfinite(value)


# --- C8 rule 5: a component may be empty -------------------------------------------


def test_an_empty_component_is_accepted() -> None:
    """C8 rule 5. A component nobody excited and that fired nothing is ordinary."""
    value = multivariate.negative_log_likelihood(
        _multi(), [np.array([1.0, 2.0]), np.array([])], 5.0
    )
    assert np.isfinite(value)


def test_all_components_empty_gives_the_compensator_alone() -> None:
    parameters = _multi()
    value = multivariate.negative_log_likelihood(
        parameters, [np.array([]), np.array([])], 5.0
    )
    assert bits(value) == bits(float(np.sum(parameters.baseline)) * 5.0)


# --- C8 rule 2: cross-component order is not defined and not required ---------------


def test_components_need_no_relation_to_each_other() -> None:
    """C8 rule 2. Nothing depends on how components interleave in storage."""
    value = multivariate.negative_log_likelihood(
        _multi(), [np.array([4.0, 4.5]), np.array([0.5, 1.0])], 6.0
    )
    assert np.isfinite(value)


# --- step 6: error mapping ----------------------------------------------------------


def test_invalid_parameters_raise_value_error() -> None:
    for baseline, excitation, decay in [
        (0.0, 0.5, 1.0),
        (-1.0, 0.5, 1.0),
        (1.0, -0.5, 1.0),
        (1.0, 0.5, 0.0),
        (float("nan"), 0.5, 1.0),
        (1.0, 0.5, float("inf")),
    ]:
        with pytest.raises(ValueError):
            univariate.Parameters(baseline, excitation, decay)


def test_a_negative_excitation_entry_raises_value_error() -> None:
    """Zero is legitimate in `d` dimensions; negative is not."""
    multivariate.Parameters(
        baseline=np.array([0.5, 0.5]),
        excitation=np.array([[0.0, 0.0], [0.0, 0.0]]),
        decay=1.0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        multivariate.Parameters(
            baseline=np.array([0.5, 0.5]),
            excitation=np.array([[0.1, -0.1], [0.0, 0.1]]),
            decay=1.0,
        )


def test_a_dimension_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError):
        multivariate.Parameters(
            baseline=np.array([0.5, 0.5, 0.5]),
            excitation=np.array([[0.1, 0.0], [0.0, 0.1]]),
            decay=1.0,
        )


def test_too_few_events_to_fit_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not enough"):
        univariate.fit(np.array([1.0, 2.0]), 5.0)
    with pytest.raises(ValueError, match="not enough"):
        multivariate.fit([np.array([1.0]), np.array([2.0])], 5.0)


def test_no_panic_reaches_the_interpreter() -> None:
    """Every documented failure surfaces as a Python exception.

    A Rust panic crossing the FFI boundary aborts the process, so the only way this
    test can report at all is if none does. `hawkes`'s library code returns `Result`
    rather than panicking (CLAUDE.md §5) and the bindings map every variant.
    """
    provocations = [
        lambda: univariate.Parameters(0.0, 0.0, 0.0),
        lambda: univariate.negative_log_likelihood(_uni(), np.array([2.0, 1.0]), 5.0),
        lambda: univariate.negative_log_likelihood(_uni(), np.array([1.0]), -1.0),
        lambda: univariate.fit(np.array([]), 5.0),
        lambda: multivariate.negative_log_likelihood(_multi(), [], 5.0),
        lambda: multivariate.negative_log_likelihood(
            _multi(3), [np.array([1.0]), np.array([2.0])], 5.0
        ),
    ]
    for provoke in provocations:
        with pytest.raises((ValueError, TypeError)):
            provoke()


def test_a_dimension_mismatch_between_parameters_and_events_raises() -> None:
    """Step 6: no panic reaches the interpreter.

    Both halves of the pair come from the caller, so a mismatch is invalid input and
    the Rust side returns an error the bindings map to ``ValueError``. Before it was
    checked at all, one direction silently returned a number and the other surfaced as
    ``pyo3_runtime.PanicException``.

    The bindings used to re-test this at the boundary because the Rust side panicked.
    That shim is gone; what this asserts is now produced by ``hawkes`` itself, so it also
    covers a Rust caller.
    """
    events_2 = [np.array([1.0]), np.array([2.0])]
    events_3 = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    with pytest.raises(ValueError, match="must agree"):
        multivariate.negative_log_likelihood(_multi(3), events_2, 5.0)
    with pytest.raises(ValueError, match="must agree"):
        multivariate.negative_log_likelihood(_multi(2), events_3, 5.0)
    with pytest.raises(ValueError, match="must agree"):
        multivariate.negative_log_likelihood_and_gradient(_multi(3), events_2, 5.0)
    with pytest.raises(ValueError, match="must agree"):
        multivariate.compensator_at_events(_multi(3), events_2, 5.0)


# --- the remaining variants, and the sweep (issue #46) ---------------------------------


def test_a_non_finite_timestamp_is_rejected() -> None:
    """C8 rule 4 cannot even be evaluated on a NaN; `hawkes` names it separately."""
    with pytest.raises(ValueError, match="not finite"):
        univariate.negative_log_likelihood(_uni(), np.array([1.0, np.nan, 3.0]), 5.0)
    with pytest.raises(ValueError, match="not finite"):
        multivariate.negative_log_likelihood(
            _multi(), [np.array([1.0]), np.array([2.0, np.nan])], 5.0
        )
    # An infinite timestamp is both non-finite and outside the window; the contract
    # does not order the two descriptions, so either message is accepted.
    with pytest.raises(ValueError, match="not finite|outside the window"):
        univariate.negative_log_likelihood(_uni(), np.array([np.inf]), 5.0)


def test_multivariate_violations_are_reported_per_component() -> None:
    """C8 is stated per component, and so is the report: the index in the message is
    the position inside the offending component."""
    with pytest.raises(ValueError, match="timestamp 6 at index 1 is outside"):
        multivariate.negative_log_likelihood(
            _multi(), [np.array([1.0, 2.0]), np.array([3.0, 6.0])], 5.0
        )
    with pytest.raises(ValueError, match="timestamp -0.5 at index 0 is outside"):
        multivariate.negative_log_likelihood(
            _multi(), [np.array([]), np.array([-0.5, 1.0])], 5.0
        )
    for horizon in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="horizon"):
            multivariate.negative_log_likelihood(_multi(), [np.array([1.0]), np.array([])], horizon)


def test_an_empty_component_list_is_rejected() -> None:
    """A process needs at least one component, whichever entry point sees it first."""
    with pytest.raises(ValueError, match="at least one component"):
        multivariate.negative_log_likelihood(_multi(), [], 5.0)
    with pytest.raises(ValueError, match="at least one component"):
        multivariate.fit([], 5.0)
    with pytest.raises(ValueError, match="at least one component"):
        multivariate.Parameters(np.array([]), np.zeros((0, 0)), 1.0)


def test_simulate_rejects_a_bad_horizon() -> None:
    """C5 applies to simulation too: the window is the caller's, and must be one."""
    for horizon in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="horizon"):
            univariate.simulate(_uni(), horizon, seed=1)
        with pytest.raises(ValueError, match="horizon"):
            multivariate.simulate(_multi(), horizon, seed=1)


def test_multivariate_fit_needs_d_plus_d_squared_plus_one_events() -> None:
    """`d = 2` needs 7 events in total, however they are distributed."""
    six = [np.array([0.5, 1.5, 2.5]), np.array([1.0, 2.0, 3.0])]
    with pytest.raises(ValueError, match="6 events is not enough"):
        multivariate.fit(six, 5.0)
    seven = [np.array([0.5, 1.5, 2.5]), np.array([1.0, 2.0, 3.0, 4.0])]
    fitted = multivariate.fit(seven, 5.0)
    assert fitted.parameters.dimension == 2


def _sorted_times(
    rng: np.random.Generator, horizon: float, on_grid: bool
) -> npt.NDArray[np.float64]:
    """Sorted timestamps in ``[0, horizon]``; on the grid, ties and both endpoints are
    common rather than incidental. Mirrors ``sorted_times`` in the Rust sweep."""
    raw = rng.uniform(0.0, 1.0, size=rng.integers(1, 40))
    if on_grid:
        raw = np.round(raw * 8.0) / 8.0
    return np.sort(raw * horizon)


def test_random_valid_realizations_are_accepted_and_one_injected_violation_is_not() -> None:
    """The randomised counterpart of the fixed cases above, through the bindings.

    Each draw injects exactly one violation into an otherwise valid realization — NaN at
    a random index, one ulp past the horizon at the last index, a negative value at
    index 0, or a value below its predecessor — so that no case depends on the order in
    which the validators run. Seeded, so a failure is reproducible.
    """
    rng = np.random.default_rng(0x5EED_46)
    rejected = 0
    for _ in range(300):
        d = int(rng.integers(1, 4))
        horizon = float(rng.uniform(0.5, 100.0))
        on_grid = bool(rng.integers(0, 2))
        events = [_sorted_times(rng, horizon, on_grid) for _ in range(d)]
        parameters = _multi(d)

        assert np.isfinite(multivariate.negative_log_likelihood(parameters, events, horizon))
        if d == 1:
            assert np.isfinite(univariate.negative_log_likelihood(_uni(), events[0], horizon))

        component = int(rng.integers(0, d))
        times = events[component].copy()
        n = len(times)
        kind = int(rng.integers(0, 4))
        if kind == 0:
            times[rng.integers(0, n)] = np.nan
            message = "not finite"
        elif kind == 1:
            times[-1] = np.nextafter(horizon, np.inf)
            message = "outside the window"
        elif kind == 2:
            times[0] = -float(rng.uniform(1e-3, 10.0))
            message = "outside the window"
        else:
            if n < 2 or times[0] <= 0.0:
                continue
            index = int(rng.integers(1, n))
            if times[index - 1] <= 0.0:
                continue
            times[index] = times[index - 1] * 0.5
            message = "ascending"
        events[component] = times

        with pytest.raises(ValueError, match=message):
            multivariate.negative_log_likelihood(parameters, events, horizon)
        if d == 1:
            with pytest.raises(ValueError, match=message):
                univariate.negative_log_likelihood(_uni(), times, horizon)
        rejected += 1
    assert rejected >= 200, f"only {rejected} injected violations were exercised"


def test_version_is_the_distribution_version() -> None:
    """``__version__`` is derived from the installed metadata, not typed a third time.
    Every wheel from 0.1.1 to 0.1.2 reported ``0.1.0`` because it was (#51)."""
    from importlib.metadata import version

    import hawkes

    assert hawkes.__version__ == version("hawkes-rs")
    assert hawkes.__version__ != "0.1.0"
