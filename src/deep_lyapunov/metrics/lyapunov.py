"""Lyapunov exponent computation for training stability analysis."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

# Module-level logger
logger = logging.getLogger(__name__)


def compute_lyapunov_exponent(
    spread_evolution: np.ndarray,
    n_epochs: int,
) -> float:
    """Compute the global Lyapunov exponent from spread evolution.

    The Lyapunov exponent measures the average rate of separation (or
    convergence) of nearby trajectories. For neural network training:

    - λ < 0: Stable training - trajectories converge
    - λ > 0: Sensitive training - trajectories diverge
    - λ ≈ 0: Neutral stability

    The exponent is computed as:
        λ = (1/T) × log(final_spread / initial_spread)

    Args:
        spread_evolution: Array of spread values at each checkpoint.
        n_epochs: Number of training epochs (time normalization).

    Returns:
        Estimated Lyapunov exponent. Negative indicates stable training.

    Example:
        >>> spread = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
        >>> lyap = compute_lyapunov_exponent(spread, n_epochs=4)
        >>> lyap < 0  # Convergent
        True
    """
    if len(spread_evolution) < 2:
        return 0.0

    initial_spread = spread_evolution[0]
    final_spread = spread_evolution[-1]

    # Handle edge cases
    if initial_spread <= 0 or final_spread <= 0:
        return 0.0

    if n_epochs <= 0:
        n_epochs = len(spread_evolution) - 1

    # Lyapunov exponent: log of ratio normalized by time
    lyapunov = np.log(final_spread / initial_spread) / n_epochs

    logger.debug(
        f"Lyapunov exponent: {lyapunov:.4f} "
        f"(initial={initial_spread:.4f}, final={final_spread:.4f}, epochs={n_epochs})"
    )

    return float(lyapunov)


def compute_local_lyapunov(
    spread_evolution: np.ndarray,
) -> np.ndarray:
    """Compute local Lyapunov exponents at each checkpoint.

    Local exponents show how stability varies during training. This is
    useful for identifying phases of instability or rapid convergence.

    The local exponent at time t is:
        λ(t) = log(spread[t+1] / spread[t])

    Args:
        spread_evolution: Array of spread values at each checkpoint.

    Returns:
        Array of local Lyapunov exponents (length n-1).

    Example:
        >>> spread = np.array([1.0, 0.8, 0.9, 0.7])
        >>> local = compute_local_lyapunov(spread)
        >>> len(local) == 3
        True
    """
    if len(spread_evolution) < 2:
        return np.array([0.0])

    # Avoid log(0) or log(negative)
    spread = np.maximum(spread_evolution, 1e-10)

    # Local exponent: log ratio of consecutive spreads
    local_lyapunov = np.log(spread[1:] / spread[:-1])

    return local_lyapunov


def compute_lyapunov_from_trajectories(
    trajectories: np.ndarray,
    reference_idx: int = 0,
) -> Tuple[float, np.ndarray]:
    """Compute Lyapunov exponent directly from trajectory data.

    This method computes the exponent by tracking divergence from a
    reference trajectory, which can be more accurate than spread-based
    methods for small perturbations.

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params)
            containing weight trajectories.
        reference_idx: Index of the reference trajectory (default: 0).

    Returns:
        Tuple of (global_lyapunov, local_lyapunov_array).

    Example:
        >>> traj = np.random.randn(10, 5, 100)  # 10 checkpoints, 5 trajectories
        >>> global_lyap, local_lyap = compute_lyapunov_from_trajectories(traj)
    """
    n_checkpoints, n_trajectories, _ = trajectories.shape

    if n_trajectories < 2:
        return 0.0, np.array([0.0])

    # Reference trajectory
    reference = trajectories[:, reference_idx, :]

    # Compute distances from reference at each checkpoint
    distances = []
    for t in range(n_checkpoints):
        dists = []
        for i in range(n_trajectories):
            if i != reference_idx:
                dist = np.linalg.norm(trajectories[t, i, :] - reference[t, :])
                dists.append(dist)
        distances.append(np.mean(dists))

    distances = np.array(distances)

    # Compute local Lyapunov from distance evolution
    local_lyap = compute_local_lyapunov(distances)

    # Global Lyapunov
    if distances[0] > 0 and distances[-1] > 0:
        global_lyap = np.log(distances[-1] / distances[0]) / (n_checkpoints - 1)
    else:
        global_lyap = 0.0

    return float(global_lyap), local_lyap
