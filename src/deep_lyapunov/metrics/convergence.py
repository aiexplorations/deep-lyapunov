"""Convergence metrics for trajectory analysis."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

# Module-level logger
logger = logging.getLogger(__name__)


def compute_convergence_ratio(
    trajectories: np.ndarray,
    n_components: int = 3,
) -> float:
    """Compute the convergence ratio from trajectory data.

    The convergence ratio measures how much the spread of trajectories
    changes during training:

    - ratio < 1: Trajectories converge (stable, reproducible)
    - ratio > 1: Trajectories diverge (sensitive to initialization)
    - ratio ≈ 1: Neutral stability

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params).
        n_components: Number of PCA components to use for spread calculation.

    Returns:
        Convergence ratio (final_spread / initial_spread).

    Example:
        >>> traj = np.random.randn(10, 5, 100)
        >>> ratio = compute_convergence_ratio(traj)
        >>> isinstance(ratio, float)
        True
    """
    n_checkpoints, n_trajectories, n_params = trajectories.shape

    logger.debug(
        f"Computing convergence ratio: {n_checkpoints} checkpoints, "
        f"{n_trajectories} trajectories, {n_params} parameters"
    )

    if n_trajectories < 2:
        logger.warning("Single trajectory provided, convergence ratio = 1.0")
        return 1.0

    # Project to PCA space for more meaningful spread calculation
    pca_traj, _ = project_to_pca(
        trajectories,
        n_components=min(n_components, n_params, n_trajectories - 1),
    )

    # Initial spread: std of first checkpoint across trajectories
    initial_points = pca_traj[0, :, :n_components]
    initial_spread = np.std(initial_points)

    # Final spread: std of last checkpoint across trajectories
    final_points = pca_traj[-1, :, :n_components]
    final_spread = np.std(final_points)

    # Avoid division by zero
    if initial_spread <= 1e-10:
        logger.debug(
            f"Very small initial spread ({initial_spread:.2e}), "
            f"final spread = {final_spread:.2e}"
        )
        return 1.0 if final_spread <= 1e-10 else float("inf")

    ratio = float(final_spread / initial_spread)
    logger.debug(
        f"Spread: initial={initial_spread:.4f}, final={final_spread:.4f}, "
        f"ratio={ratio:.4f}"
    )
    return ratio


def compute_spread_evolution(
    trajectories: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """Compute trajectory spread at each checkpoint.

    Tracks how spread evolves during training, which is useful for
    visualizing convergence/divergence patterns.

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params).
        n_components: Number of PCA components to use.

    Returns:
        Array of spread values at each checkpoint (length n_checkpoints).

    Example:
        >>> traj = np.random.randn(10, 5, 100)
        >>> spread = compute_spread_evolution(traj)
        >>> len(spread) == 10
        True
    """
    n_checkpoints, n_trajectories, n_params = trajectories.shape

    if n_trajectories < 2:
        return np.zeros(n_checkpoints)

    # Project to PCA space
    pca_traj, _ = project_to_pca(
        trajectories,
        n_components=min(n_components, n_params, n_trajectories - 1),
    )

    # Compute spread (std) at each checkpoint
    spread = np.zeros(n_checkpoints)
    for t in range(n_checkpoints):
        points = pca_traj[t, :, :n_components]
        # Use total spread across all components
        spread[t] = np.sqrt(np.sum(np.var(points, axis=0)))

    return spread


def project_to_pca(
    trajectories: np.ndarray,
    n_components: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project trajectories to PCA space.

    Projects all trajectories to a common PCA space, enabling comparison
    and visualization of weight dynamics in reduced dimensions.

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params).
        n_components: Number of PCA components to keep.

    Returns:
        Tuple of:
        - Projected trajectories of shape (n_checkpoints, n_trajectories, n_components)
        - Explained variance ratio for each component

    Example:
        >>> traj = np.random.randn(10, 5, 100)
        >>> pca_traj, var_ratio = project_to_pca(traj, n_components=5)
        >>> pca_traj.shape
        (10, 5, 5)
    """
    n_checkpoints, n_trajectories, n_params = trajectories.shape

    # Flatten for PCA fitting: (n_checkpoints * n_trajectories, n_params)
    all_points = trajectories.reshape(-1, n_params)

    # Fit PCA
    actual_components = min(n_components, n_params, all_points.shape[0])
    pca = PCA(n_components=actual_components)
    projected_flat = pca.fit_transform(all_points)

    # Reshape back to (n_checkpoints, n_trajectories, n_components)
    projected = projected_flat.reshape(n_checkpoints, n_trajectories, -1)

    return projected, pca.explained_variance_ratio_


def interpolate_trajectories(
    trajectories: np.ndarray,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """Interpolate trajectories to a common length.

    Useful when trajectories have different numbers of checkpoints.

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params).
        target_length: Desired number of checkpoints. If None, uses current length.

    Returns:
        Interpolated trajectories of shape (target_length, n_trajectories, n_params).
    """
    n_checkpoints, n_trajectories, n_params = trajectories.shape

    if target_length is None or target_length == n_checkpoints:
        return trajectories

    # Create interpolation indices
    old_indices = np.linspace(0, 1, n_checkpoints)
    new_indices = np.linspace(0, 1, target_length)

    # Interpolate each trajectory and parameter
    interpolated = np.zeros((target_length, n_trajectories, n_params))
    for i in range(n_trajectories):
        for j in range(n_params):
            interpolated[:, i, j] = np.interp(
                new_indices, old_indices, trajectories[:, i, j]
            )

    return interpolated


def compute_manifold_center(
    trajectories: np.ndarray,
) -> np.ndarray:
    """Compute the mean trajectory (manifold center).

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params).

    Returns:
        Mean trajectory of shape (n_checkpoints, n_params).
    """
    return np.mean(trajectories, axis=1)


def compute_pairwise_distances(
    points: np.ndarray,
) -> np.ndarray:
    """Compute pairwise distances between trajectory endpoints.

    Args:
        points: Array of shape (n_trajectories, n_dims).

    Returns:
        Pairwise distance matrix of shape (n_trajectories, n_trajectories).
    """
    from scipy.spatial.distance import pdist, squareform

    if len(points) < 2:
        return np.array([[0.0]])

    distances = pdist(points, metric="euclidean")
    return squareform(distances)
