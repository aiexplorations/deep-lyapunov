"""Configuration dataclasses for deep-lyapunov."""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class AnalyzerConfig:
    """Configuration for StabilityAnalyzer.

    Attributes:
        perturbation_scale: Scale of Gaussian perturbation as fraction of parameter
            standard deviation. Default 0.01 (1%).
        n_trajectories: Number of perturbed model copies to track. More trajectories
            give better statistics but increase computation. Default 5.
        n_pca_components: Number of PCA components for trajectory projection.
            Default 10.
        track_gradients: Whether to also track gradient statistics. Increases
            memory usage but provides additional insights. Default False.
        record_interval: Record weights every N training steps. Default 1 (every step).
        device: Device for computation. 'auto' uses CUDA if available. Default 'auto'.
        seed: Random seed for reproducibility. Default 42.
        verbose: Whether to print progress messages. Default True.
    """

    perturbation_scale: float = 0.01
    n_trajectories: int = 5
    n_pca_components: int = 10
    track_gradients: bool = False
    record_interval: int = 1
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.perturbation_scale <= 0:
            raise ValueError("perturbation_scale must be positive")
        if self.n_trajectories < 2:
            raise ValueError("n_trajectories must be at least 2")
        if self.n_pca_components < 1:
            raise ValueError("n_pca_components must be at least 1")
        if self.record_interval < 1:
            raise ValueError("record_interval must be at least 1")


@dataclass
class MetricsConfig:
    """Configuration for metric computation.

    Attributes:
        convergence_threshold: Convergence ratio below which training is considered
            convergent. Default 1.0.
        lyapunov_window: Number of steps for local Lyapunov estimation. Default 5.
        min_trajectory_length: Minimum trajectory length for reliable metrics.
            Default 10.
        use_robust_estimators: Use median instead of mean for robustness. Default True.
    """

    convergence_threshold: float = 1.0
    lyapunov_window: int = 5
    min_trajectory_length: int = 10
    use_robust_estimators: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization output.

    Attributes:
        figure_dpi: DPI for saved figures. Default 150.
        figure_format: Format for saved figures. Default 'png'.
        colormap: Matplotlib colormap for trajectories. Default 'viridis'.
        show_confidence_bands: Show confidence bands on plots. Default True.
    """

    figure_dpi: int = 150
    figure_format: Literal["png", "pdf", "svg"] = "png"
    colormap: str = "viridis"
    show_confidence_bands: bool = True
