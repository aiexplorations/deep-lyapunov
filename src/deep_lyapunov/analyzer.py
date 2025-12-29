"""Main StabilityAnalyzer class for neural network training stability analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

import numpy as np

from deep_lyapunov.config import AnalyzerConfig
from deep_lyapunov.metrics import (
    compute_convergence_ratio,
    compute_effective_dimensionality,
    compute_lyapunov_exponent,
    compute_participation_ratio,
    compute_spread_evolution,
    project_to_pca,
)
from deep_lyapunov.results import AnalysisResults, TrajectoryMetrics
from deep_lyapunov.tracking import TrajectoryTracker, create_perturbed_copies

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

# Module-level logger
logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """Analyze neural network training stability using Lyapunov exponents.

    StabilityAnalyzer tracks how multiple perturbed copies of a model
    evolve during training to measure stability and reproducibility.

    Two usage patterns are supported:

    1. **Automatic analysis** with `analyze()`:
       Provide a training function and the analyzer handles everything.

    2. **Manual recording** with `start_recording()`/`record_checkpoint()`:
       For custom training loops where you control when to record.

    Example (automatic):
        >>> analyzer = StabilityAnalyzer(model, perturbation_scale=0.01)
        >>> results = analyzer.analyze(
        ...     train_fn=my_train_function,
        ...     train_loader=train_loader,
        ...     n_epochs=10,
        ... )
        >>> print(f"Convergence: {results.convergence_ratio:.2f}x")

    Example (manual):
        >>> analyzer = StabilityAnalyzer(model)
        >>> analyzer.start_recording()
        >>> for epoch in range(10):
        ...     train_one_epoch(model, loader)
        ...     analyzer.record_checkpoint()
        >>> results = analyzer.compute_metrics()

    Attributes:
        model: The base PyTorch model being analyzed.
        config: Configuration settings for the analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        perturbation_scale: float = 0.01,
        n_trajectories: int = 5,
        n_pca_components: int = 10,
        track_gradients: bool = False,
        record_interval: int = 1,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        seed: int = 42,
        verbose: bool = True,
        config: Optional[AnalyzerConfig] = None,
    ) -> None:
        """Initialize the stability analyzer.

        Args:
            model: PyTorch model to analyze.
            perturbation_scale: Scale of perturbation as fraction of param std.
            n_trajectories: Number of perturbed copies to track.
            n_pca_components: Number of PCA components for projection.
            track_gradients: Whether to also track gradient statistics.
            record_interval: Record weights every N training steps.
            device: Device for computation ('auto', 'cpu', 'cuda', 'mps').
            seed: Random seed for reproducibility.
            verbose: Whether to print progress messages.
            config: Optional AnalyzerConfig to override individual params.
        """
        self._base_model = model

        # Use config if provided, otherwise build from individual params
        if config is not None:
            self._config = config
        else:
            self._config = AnalyzerConfig(
                perturbation_scale=perturbation_scale,
                n_trajectories=n_trajectories,
                n_pca_components=n_pca_components,
                track_gradients=track_gradients,
                record_interval=record_interval,
                device=device,
                seed=seed,
                verbose=verbose,
            )

        # Internal state
        self._models: List[nn.Module] = []
        self._trackers: List[TrajectoryTracker] = []
        self._is_recording = False
        self._n_checkpoints = 0

        # Resolve device
        self._device = self._resolve_device(self._config.device)

        # Log initialization
        logger.debug(
            "StabilityAnalyzer initialized: "
            f"n_trajectories={self._config.n_trajectories}, "
            f"perturbation_scale={self._config.perturbation_scale}, "
            f"device={self._device}"
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> nn.Module:
        """The base model being analyzed."""
        return self._base_model

    @property
    def config(self) -> AnalyzerConfig:
        """Current configuration."""
        return self._config

    def analyze(
        self,
        train_fn: Callable[[nn.Module, DataLoader, int], Dict[str, Any]],
        train_loader: DataLoader,
        n_epochs: int,
        test_loader: Optional[DataLoader] = None,
        **train_kwargs: Any,
    ) -> AnalysisResults:
        """Run complete stability analysis with automatic trajectory tracking.

        This method:
        1. Creates perturbed copies of the model
        2. Trains all copies using the provided training function
        3. Records weight trajectories during training
        4. Computes stability metrics

        Args:
            train_fn: Training function with signature:
                train_fn(model, train_loader, n_epochs, **kwargs) -> dict
                Should return a dict with at least 'loss' history.
            train_loader: PyTorch DataLoader for training data.
            n_epochs: Number of training epochs.
            test_loader: Optional test DataLoader for accuracy tracking.
            **train_kwargs: Additional arguments passed to train_fn.

        Returns:
            AnalysisResults containing all stability metrics.

        Example:
            >>> def train_fn(model, loader, epochs):
            ...     optimizer = Adam(model.parameters())
            ...     for epoch in range(epochs):
            ...         for batch in loader:
            ...             # ... training step ...
            ...             pass
            ...     return {'loss': [0.5, 0.3, 0.2]}
            >>> results = analyzer.analyze(train_fn, loader, n_epochs=10)
        """
        import torch

        logger.info(
            f"Starting stability analysis: {self._config.n_trajectories} trajectories, "
            f"{n_epochs} epochs"
        )
        logger.debug(
            f"Analysis config: perturbation_scale={self._config.perturbation_scale}, "
            f"seed={self._config.seed}, device={self._device}"
        )

        # Set seed for reproducibility
        torch.manual_seed(self._config.seed)

        # Create perturbed model copies
        logger.debug("Creating perturbed model copies...")
        self._models = create_perturbed_copies(
            self._base_model,
            n_copies=self._config.n_trajectories,
            scale=self._config.perturbation_scale,
            base_seed=self._config.seed,
            include_original=True,
        )

        # Move models to device
        for model in self._models:
            model.to(self._device)
        logger.debug(f"Models moved to device: {self._device}")

        # Create trackers for each model
        self._trackers = [
            TrajectoryTracker(
                model,
                track_gradients=self._config.track_gradients,
                record_interval=self._config.record_interval,
            )
            for model in self._models
        ]
        logger.debug(f"Created {len(self._trackers)} trajectory trackers")

        # Record initial state
        for tracker in self._trackers:
            tracker.capture()
        logger.debug("Captured initial weight state for all trajectories")

        # Train each model
        training_metrics = []
        for i, (model, tracker) in enumerate(zip(self._models, self._trackers)):
            logger.info(f"Training trajectory {i + 1}/{self._config.n_trajectories}...")

            # Train with checkpoint recording
            result = self._train_with_recording(
                model,
                tracker,
                train_fn,
                train_loader,
                n_epochs,
                test_loader=test_loader,
                **train_kwargs,
            )
            training_metrics.append(result)

            # Log training result summary
            if result.get("loss"):
                final_loss = result["loss"][-1] if result["loss"] else None
                logger.debug(
                    f"Trajectory {i + 1} complete: final_loss={final_loss:.4f}"
                    if final_loss
                    else f"Trajectory {i + 1} complete"
                )

        self._n_checkpoints = self._trackers[0].n_snapshots
        logger.debug(f"Training complete: {self._n_checkpoints} checkpoints recorded")

        # Compute metrics
        logger.info("Computing stability metrics...")
        return self.compute_metrics(training_metrics)

    def _train_with_recording(
        self,
        model: nn.Module,
        tracker: TrajectoryTracker,
        train_fn: Callable,
        train_loader: DataLoader,
        n_epochs: int,
        test_loader: Optional[DataLoader] = None,
        **train_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train a model while recording checkpoints.

        This wraps the user's training function to capture weight states
        at regular intervals.
        """
        import torch

        # Simple approach: train and capture after each epoch
        history = {"loss": [], "accuracy": []}

        # We need to hook into the training loop
        # For simplicity, we'll call train_fn for 1 epoch at a time
        for epoch in range(n_epochs):
            result = train_fn(model, train_loader, 1, **train_kwargs)

            # Extract metrics
            if "loss" in result:
                loss = (
                    result["loss"][-1]
                    if isinstance(result["loss"], list)
                    else result["loss"]
                )
                history["loss"].append(loss)
            if "accuracy" in result:
                acc = (
                    result["accuracy"][-1]
                    if isinstance(result["accuracy"], list)
                    else result["accuracy"]
                )
                history["accuracy"].append(acc)
            elif "test_acc" in result:
                acc = (
                    result["test_acc"][-1]
                    if isinstance(result["test_acc"], list)
                    else result["test_acc"]
                )
                history["accuracy"].append(acc)

            # Capture checkpoint
            tracker.capture()

            # Log epoch progress at DEBUG level
            loss_str = f"loss={history['loss'][-1]:.4f}" if history["loss"] else ""
            acc_str = (
                f", acc={history['accuracy'][-1]:.4f}" if history["accuracy"] else ""
            )
            logger.debug(f"  Epoch {epoch + 1}/{n_epochs}: {loss_str}{acc_str}")

        return history

    def start_recording(self) -> None:
        """Start manual recording mode.

        Call this before your training loop, then call `record_checkpoint()`
        after each epoch or at desired intervals.

        Example:
            >>> analyzer.start_recording()
            >>> for epoch in range(10):
            ...     train_one_epoch(model, loader)
            ...     analyzer.record_checkpoint()
            >>> results = analyzer.compute_metrics()
        """
        import torch

        if self._is_recording:
            raise RuntimeError("Already recording. Call reset() first.")

        logger.info(
            f"Starting manual recording: {self._config.n_trajectories} trajectories, "
            f"perturbation_scale={self._config.perturbation_scale}"
        )

        torch.manual_seed(self._config.seed)

        # Create perturbed copies
        self._models = create_perturbed_copies(
            self._base_model,
            n_copies=self._config.n_trajectories,
            scale=self._config.perturbation_scale,
            base_seed=self._config.seed,
            include_original=True,
        )
        logger.debug(f"Created {len(self._models)} perturbed model copies")

        # Move to device
        for model in self._models:
            model.to(self._device)

        # Create trackers
        self._trackers = [
            TrajectoryTracker(
                model,
                track_gradients=self._config.track_gradients,
                record_interval=self._config.record_interval,
            )
            for model in self._models
        ]

        # Record initial state
        for tracker in self._trackers:
            tracker.capture()

        self._is_recording = True
        self._n_checkpoints = 1

        n_params = self._trackers[0].n_parameters
        logger.info(f"Recording started: tracking {n_params:,} parameters")

    def record_checkpoint(self) -> None:
        """Record current weight state of all model copies.

        Call this after each training epoch or at desired intervals.

        Raises:
            RuntimeError: If not in recording mode.
        """
        if not self._is_recording:
            raise RuntimeError("Not recording. Call start_recording() first.")

        for tracker in self._trackers:
            tracker.capture()

        self._n_checkpoints += 1
        logger.debug(f"Checkpoint {self._n_checkpoints} recorded")

    def get_models(self) -> List[nn.Module]:
        """Get the list of perturbed model copies.

        Use this to access the models for custom training loops.

        Returns:
            List of model copies (first is unperturbed original).
        """
        if not self._models:
            raise RuntimeError("Models not created. Call start_recording() first.")
        return self._models

    def compute_metrics(
        self,
        training_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> AnalysisResults:
        """Compute stability metrics from recorded trajectories.

        Args:
            training_metrics: Optional per-trajectory training metrics
                (loss, accuracy history).

        Returns:
            AnalysisResults with all computed stability metrics.

        Raises:
            RuntimeError: If no trajectories have been recorded.
        """
        if not self._trackers or self._trackers[0].n_snapshots < 2:
            raise RuntimeError(
                "Insufficient trajectory data. Record at least 2 checkpoints."
            )

        # Collect trajectories
        trajectories = self._collect_trajectories()

        # Compute core metrics
        convergence_ratio = compute_convergence_ratio(
            trajectories,
            n_components=min(3, self._config.n_pca_components),
        )

        spread_evolution = compute_spread_evolution(
            trajectories,
            n_components=min(3, self._config.n_pca_components),
        )

        n_epochs = self._n_checkpoints - 1
        lyapunov = compute_lyapunov_exponent(spread_evolution, n_epochs)

        behavior: Literal["convergent", "divergent"] = (
            "convergent" if convergence_ratio < 1.0 else "divergent"
        )

        # PCA projection
        pca_trajectories, pca_variance = project_to_pca(
            trajectories,
            n_components=self._config.n_pca_components,
        )

        # Effective dimensionality
        eff_dim_array = compute_effective_dimensionality(trajectories)
        effective_dim = float(np.mean(eff_dim_array))

        # Per-trajectory metrics
        trajectory_metrics = self._compute_trajectory_metrics(training_metrics)

        # Get metadata
        n_params = self._trackers[0].n_parameters

        results = AnalysisResults(
            convergence_ratio=float(convergence_ratio),
            lyapunov=float(lyapunov),
            behavior=behavior,
            spread_evolution=spread_evolution,
            pca_trajectories=pca_trajectories,
            pca_explained_variance=pca_variance,
            effective_dimensionality=effective_dim,
            trajectory_metrics=trajectory_metrics,
            n_trajectories=self._config.n_trajectories,
            n_checkpoints=self._n_checkpoints,
            n_parameters=n_params,
        )

        # Log comprehensive results
        logger.info(
            f"Analysis complete: convergence_ratio={convergence_ratio:.3f}x, "
            f"lyapunov={lyapunov:.4f}, behavior={behavior}"
        )
        logger.debug(
            f"Additional metrics: effective_dim={effective_dim:.2f}, "
            f"n_checkpoints={self._n_checkpoints}, n_params={n_params:,}"
        )

        # Log interpretation
        if behavior == "convergent":
            logger.info(
                "Interpretation: Training is STABLE - different initializations "
                "converge to similar solutions"
            )
        else:
            logger.info(
                "Interpretation: Training is SENSITIVE - small differences in "
                "initialization lead to different solutions"
            )

        return results

    def _collect_trajectories(self) -> np.ndarray:
        """Collect trajectories from all trackers into single array.

        Returns:
            Array of shape (n_checkpoints, n_trajectories, n_params).
        """
        # Get trajectories from trackers
        raw_trajectories = [tracker.get_trajectory() for tracker in self._trackers]

        # Ensure all same length (use minimum)
        min_len = min(len(t) for t in raw_trajectories)
        trajectories = np.array([t[:min_len] for t in raw_trajectories])

        # Transpose to (n_checkpoints, n_trajectories, n_params)
        return trajectories.transpose(1, 0, 2)

    def _compute_trajectory_metrics(
        self,
        training_metrics: Optional[List[Dict[str, Any]]],
    ) -> List[TrajectoryMetrics]:
        """Compute per-trajectory metrics."""
        metrics = []

        for i, tracker in enumerate(self._trackers):
            stats = tracker.get_statistics()

            tm = TrajectoryMetrics(
                trajectory_id=i,
                path_length=stats["path_length"],
                velocity_mean=stats["velocity_mean"],
                velocity_std=stats["velocity_std"],
            )

            # Add training metrics if available
            if training_metrics and i < len(training_metrics):
                tm_dict = training_metrics[i]
                if "loss" in tm_dict and tm_dict["loss"]:
                    tm.final_loss = tm_dict["loss"][-1]
                if "accuracy" in tm_dict and tm_dict["accuracy"]:
                    tm.final_accuracy = tm_dict["accuracy"][-1]

            metrics.append(tm)

        return metrics

    def reset(self) -> None:
        """Reset the analyzer state for a new analysis.

        Clears all recorded trajectories and model copies.
        """
        old_checkpoints = self._n_checkpoints
        self._models = []
        self._trackers = []
        self._is_recording = False
        self._n_checkpoints = 0

        logger.info(f"Analyzer reset (cleared {old_checkpoints} checkpoints)")
