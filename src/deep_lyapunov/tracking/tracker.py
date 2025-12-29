"""Trajectory tracker for capturing weight states during training."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn

# Module-level logger
logger = logging.getLogger(__name__)


class TrajectoryTracker:
    """Captures weight states during neural network training.

    This class records weight snapshots at configurable intervals,
    storing them for later analysis of training dynamics.

    Example:
        >>> tracker = TrajectoryTracker(model)
        >>> for epoch in range(10):
        ...     train_one_epoch(model)
        ...     tracker.capture()
        >>> trajectory = tracker.get_trajectory()
        >>> print(f"Shape: {trajectory.shape}")  # (10, n_params)

    Attributes:
        model: The PyTorch model to track.
        track_gradients: Whether to also capture gradient statistics.
        record_interval: Capture every N calls to `capture()`.
    """

    def __init__(
        self,
        model: nn.Module,
        track_gradients: bool = False,
        record_interval: int = 1,
    ) -> None:
        """Initialize the trajectory tracker.

        Args:
            model: PyTorch model to track.
            track_gradients: Whether to track gradient statistics.
            record_interval: Record weights every N calls to capture().
        """
        import torch.nn as nn

        self._model = model
        self._track_gradients = track_gradients
        self._record_interval = record_interval

        # Storage
        self._weight_snapshots: List[np.ndarray] = []
        self._gradient_snapshots: List[np.ndarray] = []
        self._capture_count = 0

        # Cache parameter count
        self._n_params = self._count_parameters()

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def _extract_weight_vector(self) -> np.ndarray:
        """Extract all model weights as a single flattened vector.

        Returns:
            1D numpy array containing all trainable parameters.
        """
        import torch

        weights = []
        for param in self._model.parameters():
            if param.requires_grad:
                weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(weights)

    def _extract_gradient_vector(self) -> Optional[np.ndarray]:
        """Extract all gradients as a single flattened vector.

        Returns:
            1D numpy array of gradients, or None if gradients not available.
        """
        gradients = []
        has_grads = False

        for param in self._model.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    gradients.append(param.grad.detach().cpu().numpy().flatten())
                    has_grads = True
                else:
                    # Fill with zeros if gradient not computed
                    gradients.append(np.zeros(param.numel()))

        return np.concatenate(gradients) if has_grads else None

    def capture(self) -> None:
        """Capture current weight state.

        Records a snapshot of all trainable parameters. If track_gradients
        is enabled and gradients are available, also records gradient state.
        """
        self._capture_count += 1

        if self._capture_count % self._record_interval != 0:
            return

        # Capture weights
        weight_vector = self._extract_weight_vector()
        self._weight_snapshots.append(weight_vector)

        # Capture gradients if enabled
        if self._track_gradients:
            grad_vector = self._extract_gradient_vector()
            if grad_vector is not None:
                self._gradient_snapshots.append(grad_vector)

    def get_trajectory(self) -> np.ndarray:
        """Get the weight trajectory as a 2D array.

        Returns:
            Array of shape (n_snapshots, n_params) containing weight states
            at each captured checkpoint.

        Raises:
            ValueError: If no snapshots have been captured.
        """
        if not self._weight_snapshots:
            raise ValueError("No snapshots captured. Call capture() during training.")
        return np.array(self._weight_snapshots)

    def get_gradient_trajectory(self) -> Optional[np.ndarray]:
        """Get the gradient trajectory as a 2D array.

        Returns:
            Array of shape (n_snapshots, n_params) containing gradient states,
            or None if gradients were not tracked.
        """
        if not self._gradient_snapshots:
            return None
        return np.array(self._gradient_snapshots)

    def get_latest_weights(self) -> Optional[np.ndarray]:
        """Get the most recently captured weight vector.

        Returns:
            1D array of weights, or None if no captures yet.
        """
        if not self._weight_snapshots:
            return None
        return self._weight_snapshots[-1]

    @property
    def n_snapshots(self) -> int:
        """Number of captured snapshots."""
        return len(self._weight_snapshots)

    @property
    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return self._n_params

    @property
    def model(self) -> nn.Module:
        """The tracked model."""
        return self._model

    def reset(self) -> None:
        """Clear all captured snapshots."""
        self._weight_snapshots.clear()
        self._gradient_snapshots.clear()
        self._capture_count = 0

    def get_statistics(self) -> Dict[str, float]:
        """Compute basic statistics about the trajectory.

        Returns:
            Dictionary with trajectory statistics including path length,
            mean velocity, and velocity standard deviation.
        """
        if len(self._weight_snapshots) < 2:
            return {
                "path_length": 0.0,
                "velocity_mean": 0.0,
                "velocity_std": 0.0,
            }

        trajectory = self.get_trajectory()

        # Compute displacements between consecutive snapshots
        displacements = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(displacements, axis=1)

        return {
            "path_length": float(np.sum(distances)),
            "velocity_mean": float(np.mean(distances)),
            "velocity_std": float(np.std(distances)),
        }
