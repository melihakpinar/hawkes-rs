"""The array policy of ``docs/python-array-handling.md``, enforced (M3 step 5).

numpy has a default for each of these — a silent cast, a silent copy, a silent
reinterpretation of strides. Each test names the section of that document it enforces,
so a behaviour change has to be a documented decision rather than a shrug.
"""

from __future__ import annotations

import numpy as np
import pytest

from hawkes import multivariate, univariate

from .helpers import bits


def _uni() -> univariate.Parameters:
    return univariate.Parameters(0.5, 0.4, 1.0)


# Deliberately asymmetric: a transposition is undetectable on a symmetric matrix.
ASYMMETRIC = np.array([[0.05, 0.60], [0.02, 0.10]])
BASELINE = np.array([0.30, 0.20])
EVENTS = [np.array([1.0, 2.5, 4.0]), np.array([0.5, 3.0])]
HORIZON = 6.0


# --- §1: dtype -- float64 only ------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64, np.int32])
def test_non_float64_arrays_are_rejected(dtype: type) -> None:
    """§1. No cast is performed, not even a widening one.

    A ``float32`` timestamp array cannot separate events near a realistic horizon and
    silently manufactures ties, which invalidates the maximum-likelihood asymptotics.
    Widening it here would hide that the precision was already gone.
    """
    times = np.array([1.0, 2.0, 3.0], dtype=dtype)
    with pytest.raises(TypeError):
        univariate.negative_log_likelihood(_uni(), times, 5.0)


def test_non_float64_components_are_rejected_with_a_usable_message() -> None:
    """§1, and the message has to say what to do about it."""
    events = [np.array([1.0], dtype=np.float32), np.array([2.0])]
    with pytest.raises(TypeError, match="float64"):
        multivariate.negative_log_likelihood(
            multivariate.Parameters(BASELINE, ASYMMETRIC, 1.0), events, 5.0
        )


def test_a_python_list_is_rejected() -> None:
    """§1. Accepting lists would mean inferring a dtype, and ``[1, 2]`` infers
    ``int64``, which rule 1 then refuses for a reason invisible in the source."""
    with pytest.raises(TypeError):
        univariate.negative_log_likelihood(_uni(), [1.0, 2.0], 5.0)  # type: ignore[arg-type]


def test_float64_is_accepted_however_it_was_produced() -> None:
    """§1, the accepting half. Nothing about the array's provenance matters."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.asarray([1, 2, 3], dtype=np.float64)
    c = np.array([1.0, 2.0, 3.0], dtype=np.float32).astype(np.float64)
    reference = univariate.negative_log_likelihood(_uni(), a, 5.0)
    for other in (b, c):
        assert bits(univariate.negative_log_likelihood(_uni(), other, 5.0)) == bits(reference)


# --- §2: contiguity ------------------------------------------------------------------


def test_a_non_contiguous_array_is_accepted_and_gives_the_same_answer() -> None:
    """§2. Accepted and copied; a copy changes no value."""
    dense = np.array([1.0, 99.0, 2.0, 99.0, 3.0, 99.0])
    strided = dense[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    contiguous = np.ascontiguousarray(strided)
    assert bits(univariate.negative_log_likelihood(_uni(), strided, 5.0)) == bits(
        univariate.negative_log_likelihood(_uni(), contiguous, 5.0)
    )


def test_inputs_are_never_modified() -> None:
    """§4's promise: a caller may pass an array and keep using it."""
    times = np.array([1.0, 2.0, 3.0])
    before = times.copy()
    univariate.fit(times, 5.0)
    univariate.negative_log_likelihood(_uni(), times, 5.0)
    assert np.array_equal(times, before)


# --- §3: memory order -- the transposition hazard -------------------------------------


def test_a_fortran_ordered_excitation_matrix_is_not_transposed() -> None:
    """§3, and the most dangerous case in the document.

    ``hawkes`` stores excitation row-major and reads ``excitation[i*d + j]`` as "j excites
    i" (C6). Handing Rust the raw buffer of an F-ordered array would deliver **the
    transpose** — a plausible-looking matrix that is wrong and undetectable on symmetric
    input.

    ``np.asfortranarray(a)`` and ``a`` differ only in layout, never in ``a[i, j]``, so
    the two must give the same answer, and it must be the answer for ``a`` rather than
    for ``a.T``.
    """
    c_order = np.ascontiguousarray(ASYMMETRIC)
    f_order = np.asfortranarray(ASYMMETRIC)
    assert f_order.flags["F_CONTIGUOUS"] and not f_order.flags["C_CONTIGUOUS"]
    assert np.array_equal(c_order, f_order), "same logical content, different layout"

    from_c = multivariate.negative_log_likelihood(
        multivariate.Parameters(BASELINE, c_order, 1.0), EVENTS, HORIZON
    )
    from_f = multivariate.negative_log_likelihood(
        multivariate.Parameters(BASELINE, f_order, 1.0), EVENTS, HORIZON
    )
    from_transpose = multivariate.negative_log_likelihood(
        multivariate.Parameters(BASELINE, np.ascontiguousarray(ASYMMETRIC.T), 1.0),
        EVENTS,
        HORIZON,
    )

    assert bits(from_f) == bits(from_c), (
        "an F-ordered excitation matrix was read by stride rather than by index, "
        "which transposes it"
    )
    assert bits(from_transpose) != bits(from_c), (
        "the test matrix is too close to symmetric for a transposition to be visible; "
        "this test would pass either way"
    )


def test_a_fortran_ordered_matrix_round_trips_through_the_getter() -> None:
    """§3 and §6. What comes back is what went in, logically, in C order."""
    parameters = multivariate.Parameters(BASELINE, np.asfortranarray(ASYMMETRIC), 1.0)
    returned = parameters.excitation
    assert np.array_equal(returned, ASYMMETRIC)
    assert returned.flags["C_CONTIGUOUS"]


# --- §4: writability -------------------------------------------------------------------


def test_a_read_only_array_is_accepted() -> None:
    """§4. Read-only arrays come from ``np.frombuffer``, mmap and shared memory, and
    the bindings only read."""
    times = np.array([1.0, 2.0, 3.0])
    times.flags.writeable = False
    assert not times.flags["WRITEABLE"]
    value = univariate.negative_log_likelihood(_uni(), times, 5.0)
    assert np.isfinite(value)

    excitation = np.array(ASYMMETRIC)
    excitation.flags.writeable = False
    baseline = np.array(BASELINE)
    baseline.flags.writeable = False
    parameters = multivariate.Parameters(baseline, excitation, 1.0)
    assert np.isfinite(multivariate.negative_log_likelihood(parameters, EVENTS, HORIZON))


# --- §5: shape and dimensionality ------------------------------------------------------


def test_wrong_dimensionality_is_rejected() -> None:
    """§5. No broadcasting, no squeezing, no promoting a scalar."""
    with pytest.raises(TypeError):
        univariate.negative_log_likelihood(_uni(), np.array([[1.0, 2.0]]), 5.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        multivariate.Parameters(BASELINE, np.array([0.1, 0.2, 0.3, 0.4]), 1.0)


def test_a_non_square_excitation_matrix_is_rejected() -> None:
    """§5."""
    with pytest.raises(ValueError, match="square"):
        multivariate.Parameters(BASELINE, np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]), 1.0)


# --- §6: returned arrays are freshly owned ---------------------------------------------


def test_returned_arrays_belong_to_the_caller() -> None:
    """§6. The behaviour is promised, not the ``OWNDATA`` flag.

    A matrix built row by row owns its buffer; an array handed over from a Rust ``Vec``
    is adopted by numpy through a base object and reports ``OWNDATA == False`` while the
    memory is still numpy's. Both are safe, so the assertions here are about what a
    caller can actually rely on: it is writable, mutating it changes nothing else, and
    it does not depend on a Rust value outliving it.
    """
    parameters = multivariate.Parameters(BASELINE, ASYMMETRIC, 1.0)
    simulated = univariate.simulate(_uni(), 100.0, seed=1)
    for array in (parameters.baseline, parameters.excitation, simulated):
        assert array.flags["WRITEABLE"]

    # Mutating a result must not affect the object it came from, nor a later read.
    before = parameters.excitation.copy()
    scratch = parameters.excitation
    scratch[0, 0] = 999.0
    assert np.array_equal(parameters.excitation, before)

    # The source arrays the caller passed in are likewise untouched.
    assert np.array_equal(ASYMMETRIC, np.array([[0.05, 0.60], [0.02, 0.10]]))

    # A result outlives the object that produced it.
    del parameters
    assert np.isfinite(scratch).all()
    assert np.isfinite(simulated).all()
