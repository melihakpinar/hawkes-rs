"""Simulate a three-component Hawkes process, fit it, and read the result back.

Run with ``python examples/fit_multivariate.py``. Type-checked by ``mypy --strict``.
"""

from __future__ import annotations

import numpy as np

from hawkes import multivariate


def main() -> None:
    truth = multivariate.Parameters(
        baseline=np.array([0.30, 0.20, 0.50]),
        # excitation[i][j] is "j excites i": the FIRST index is the excited component.
        excitation=np.array(
            [
                [0.10, 0.45, 0.05],
                [0.05, 0.10, 0.30],
                [0.20, 0.05, 0.10],
            ]
        ),
        decay=1.4,
    )
    horizon = 4000.0

    print(f"spectral radius {truth.branching_ratio_spectral_radius():.6f}")
    print(f"stationary      {truth.is_stationary()}")
    mean = truth.stationary_mean_intensity()
    if mean is None:
        raise SystemExit("non-stationary parameters have no stationary mean intensity")
    print(f"mean intensity  {np.round(mean, 6)}")

    events = multivariate.simulate(truth, horizon=horizon, seed=3)
    print(f"simulated       {[len(component) for component in events]} events")

    fitted = multivariate.fit(events, horizon=horizon)
    print(f"converged       {fitted.converged} after {fitted.iterations} iterations")
    print(f"gradient norm   {fitted.gradient_norm:.3e}")
    print(f"baseline        {np.round(fitted.parameters.baseline, 4)}")
    print("excitation      (row i excited, column j exciting)")
    for row in np.round(fitted.parameters.excitation, 4):
        print(f"                {row}")
    print(f"spectral radius {fitted.branching_ratio_spectral_radius():.6f}")

    # Residual analysis: per component, the compensated inter-arrival times should be
    # Exp(1) if the parameters are right. Pooling them would let an error in one
    # component be masked by another.
    compensators = multivariate.compensator_at_events(fitted.parameters, events, horizon)
    for index, values in enumerate(compensators):
        residuals = np.diff(np.concatenate([np.zeros(1), values]))
        print(f"component {index}     mean residual {residuals.mean():.4f} (Exp(1) has mean 1)")


if __name__ == "__main__":
    main()
