"""Prove the pinned `tick` environment can actually fit a Hawkes model.

This is the acceptance check for the container: it imports tick, simulates a small
univariate Hawkes process with a fixed seed, fits it by maximum likelihood, and
prints the recovered parameters. If this runs, the oracle is usable.

Run with:  docker run --rm --platform=linux/amd64 hawk-tick:0.8.0.2
"""

import sys

import numpy as np
from tick.hawkes import ModelHawkesExpKernLogLik, SimuHawkesExpKernels
from tick.prox import ProxPositive
from tick.solver import AGD

import tick

# Truth for the smoke model. Under tick's kernel convention
# phi_ij(t) = alpha_ij * beta_ij * exp(-beta_ij * t), `adjacency` is already the
# kernel integral, so the branching ratio is `adjacency` itself and stationarity
# requires its spectral radius to be < 1.
# tick/hawkes/model/model_hawkes_expkern_loglik.py:41-43
BASELINE = np.array([0.5])
ADJACENCY = np.array([[0.2]])
DECAY = 1.5
END_TIME = 500.0
SEED = 1234


def main() -> int:
    print(f"python : {sys.version.split()[0]}")
    print(f"tick   : {tick.__version__}")
    print(f"numpy  : {np.__version__}")

    simulator = SimuHawkesExpKernels(
        adjacency=ADJACENCY,
        decays=np.full_like(ADJACENCY, DECAY),
        baseline=BASELINE,
        end_time=END_TIME,
        seed=SEED,
        verbose=False,
    )
    simulator.simulate()
    events = [timestamps.copy() for timestamps in simulator.timestamps]
    print(f"simulated: {[len(t) for t in events]} events on [0, {END_TIME}]")
    print(f"spectral radius: {simulator.spectral_radius()}")

    model = ModelHawkesExpKernLogLik(decay=DECAY)
    # end_times must be passed explicitly. When it is None, tick infers the window
    # as max(events) (tick/hawkes/model/base/model_hawkes.py:88-91), which biases
    # the baseline upward because the trailing dead time is discarded.
    model.fit([events], end_times=END_TIME)

    # `gofit="likelihood"` needs a positivity constraint: the C++ model rejects a
    # negative total influence, so the solver must be given ProxPositive rather
    # than the ProxZero that HawkesExpKern(penalty="none") installs.
    solver = AGD(max_iter=1000, tol=1e-12, verbose=False, step=1e-1)
    solver.set_model(model).set_prox(ProxPositive())
    fitted = solver.solve(np.array([0.1, 0.1]))

    print(f"true   baseline={BASELINE.tolist()} adjacency={ADJACENCY.ravel().tolist()}")
    print(f"fitted baseline=[{fitted[0]!r}] adjacency=[{fitted[1]!r}]")
    print(f"loss at fitted: {model.loss(fitted)!r}")
    print(f"loss at truth : {model.loss(np.concatenate([BASELINE, ADJACENCY.ravel()]))!r}")

    if not np.isfinite(fitted).all():
        print("FAIL: fit did not produce finite parameters")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
