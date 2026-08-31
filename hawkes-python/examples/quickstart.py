"""The example in the README. Kept here so CI type-checks and runs it.

Simulates a univariate exponential-kernel Hawkes process, fits it back, and reports
the stationarity diagnostic.
"""

from __future__ import annotations

import numpy as np

from hawkes import univariate


def main() -> None:
    truth = univariate.Parameters(baseline=0.5, excitation=0.6, decay=1.0)

    # `horizon` is supplied, never inferred from the events. An inferred window
    # silently biases the baseline; see docs/derivations/conventions.md C5.
    horizon = 20_000.0
    times = univariate.simulate(truth, horizon, seed=7)

    fit = univariate.fit(times, horizon)
    print(f"{len(times)} events on [0, {horizon:g}]")
    print(f"  baseline   {fit.parameters.baseline:.4f}  (true {truth.baseline})")
    print(f"  excitation {fit.parameters.excitation:.4f}  (true {truth.excitation})")
    print(f"  decay      {fit.parameters.decay:.4f}  (true {truth.decay})")
    print(f"  converged={fit.converged} iterations={fit.iterations}")

    # Stationarity is a diagnostic on the result, not a constraint during fitting:
    # a non-stationary fit is a real finding about the data (CLAUDE.md §6).
    ratio = fit.branching_ratio()
    print(f"  branching ratio {ratio:.4f} -> stationary={fit.is_stationary()}")


if __name__ == "__main__":
    main()
