"""One-to-one anonymous detector event matching, independent of SUMO.

Matching maximizes coverage within a timing tolerance before minimizing absolute
error. Duplicate timestamps are distinct events; no persistent vehicle identity
is implied by an anonymous match.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def optimal_pairs(raw, sim, tolerance=120):
    """Maximum cardinality within tolerance, then minimum total absolute error.

    Return index pairs; indices distinguish repeated equal-time events. No reuse.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Timing tolerance must be finite and nonnegative.")
    raw = np.asarray(raw, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if (
        raw.ndim != 1
        or sim.ndim != 1
        or not np.isfinite(raw).all()
        or not np.isfinite(sim).all()
    ):
        raise ValueError("Event timestamps must be finite one-dimensional sequences.")
    if raw.size == 0 or sim.size == 0:
        return []
    delta = sim[None, :] - raw[:, None]
    distance = abs(delta)
    n, m = distance.shape
    unmatched = (n + 1) * (tolerance + 1)
    cost = np.concatenate(
        [
            np.where(distance <= tolerance, distance, unmatched * (n + 2)),
            np.full((n, n), unmatched),
        ],
        axis=1,
    )
    rr, cc = linear_sum_assignment(cost)
    return [
        (int(i), int(j), float(delta[i, j]))
        for i, j in zip(rr, cc)
        if j < m and distance[i, j] <= tolerance
    ]
