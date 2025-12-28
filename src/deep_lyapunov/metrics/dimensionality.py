"""Effective dimensionality metrics for weight dynamics analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_effective_dimensionality(
    trajectories: np.ndarray,
    window_size: Optional[int] = None,
) -> np.ndarray:
    """Compute effective dimensionality over training using participation ratio.

    The effective dimensionality measures how many dimensions are
    "effectively" used in the weight dynamics. This is computed using
    the participation ratio of covariance matrix eigenvalues.

    A high effective dimensionality indicates complex, high-dimensional
    dynamics, while low values suggest the learning happens in a
    low-dimensional subspace.

    Args:
        trajectories: Array of shape (n_checkpoints, n_trajectories, n_params).
        window_size: Size of sliding window for local analysis. If None,
            uses min(50, n_checkpoints // 4).

    Returns:
        Array of effective dimensionality values at each window position.

    Example:
        >>> traj = np.random.randn(100, 5, 50)
        >>> eff_dim = compute_effective_dimensionality(traj)
        >>> eff_dim.mean() > 0
        True
    """
    n_checkpoints, n_trajectories, n_params = trajectories.shape

    if window_size is None:
        window_size = min(50, max(10, n_checkpoints // 4))

    effective_dims = []

    # Sliding window analysis
    step_size = max(1, window_size // 2)
    for i in range(0, n_checkpoints - window_size + 1, step_size):
        # Get window of data: flatten trajectories in window
        window = trajectories[i : i + window_size, :, :]
        # Reshape to (window_size * n_trajectories, n_params)
        window_flat = window.reshape(-1, n_params)

        eff_dim = compute_participation_ratio(window_flat)
        effective_dims.append(eff_dim)

    if not effective_dims:
        # If window too large, compute global participation ratio
        all_points = trajectories.reshape(-1, n_params)
        return np.array([compute_participation_ratio(all_points)])

    return np.array(effective_dims)


def compute_participation_ratio(
    data: np.ndarray,
) -> float:
    """Compute participation ratio from data matrix.

    The participation ratio (PR) measures the effective number of
    dimensions used by a distribution:

        PR = (Σλᵢ)² / Σλᵢ²

    where λᵢ are eigenvalues of the covariance matrix.

    Properties:
    - PR = 1 when all variance is in one dimension
    - PR = n when variance is equally distributed across n dimensions
    - PR is always between 1 and min(n_samples, n_features)

    Args:
        data: Array of shape (n_samples, n_features).

    Returns:
        Participation ratio (effective dimensionality).

    Example:
        >>> # 1D data has PR ≈ 1
        >>> data_1d = np.column_stack([np.random.randn(100), np.zeros(100)])
        >>> pr = compute_participation_ratio(data_1d)
        >>> pr < 2
        True
    """
    n_samples, n_features = data.shape

    if n_samples < 2 or n_features < 1:
        return 1.0

    # Compute covariance matrix
    # Use rowvar=False because samples are in rows
    try:
        cov = np.cov(data, rowvar=False)
    except Exception:
        return 1.0

    # Handle scalar covariance (single feature)
    if cov.ndim == 0:
        return 1.0

    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 1.0

    # Ensure non-negative (numerical stability)
    eigenvalues = np.maximum(eigenvalues, 0)

    # Participation ratio
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues**2)

    if sum_eig_sq <= 0:
        return 1.0

    pr = (sum_eig**2) / sum_eig_sq

    return float(pr)


def compute_cumulative_variance(
    explained_variance_ratio: np.ndarray,
) -> np.ndarray:
    """Compute cumulative explained variance.

    Args:
        explained_variance_ratio: Variance ratio for each PCA component.

    Returns:
        Cumulative variance ratio array.

    Example:
        >>> var_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        >>> cum_var = compute_cumulative_variance(var_ratio)
        >>> cum_var[-1]
        1.0
    """
    return np.cumsum(explained_variance_ratio)


def find_intrinsic_dimension(
    explained_variance_ratio: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """Find number of components needed to explain threshold variance.

    Args:
        explained_variance_ratio: Variance ratio for each PCA component.
        threshold: Variance threshold (default 0.95 = 95%).

    Returns:
        Number of components needed.

    Example:
        >>> var_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        >>> n_dims = find_intrinsic_dimension(var_ratio, threshold=0.95)
        >>> n_dims
        3
    """
    cumulative = compute_cumulative_variance(explained_variance_ratio)
    idx = np.searchsorted(cumulative, threshold) + 1
    return min(int(idx), len(explained_variance_ratio))


def compute_local_dimensionality(
    trajectory: np.ndarray,
    k_neighbors: int = 10,
) -> np.ndarray:
    """Compute local intrinsic dimensionality along a trajectory.

    Uses a k-nearest neighbor approach to estimate local dimensionality
    at each point in the trajectory.

    Args:
        trajectory: Single trajectory of shape (n_checkpoints, n_params).
        k_neighbors: Number of neighbors for local estimation.

    Returns:
        Array of local dimensionality estimates.
    """
    n_checkpoints = len(trajectory)

    if n_checkpoints < k_neighbors + 1:
        return np.ones(n_checkpoints)

    local_dims = []

    for i in range(n_checkpoints):
        # Get k nearest neighbors (use temporal neighbors)
        start = max(0, i - k_neighbors // 2)
        end = min(n_checkpoints, i + k_neighbors // 2 + 1)
        local_window = trajectory[start:end]

        if len(local_window) >= 3:
            dim = compute_participation_ratio(local_window)
        else:
            dim = 1.0

        local_dims.append(dim)

    return np.array(local_dims)
