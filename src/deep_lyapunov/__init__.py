"""
deep-lyapunov: Analyze neural network training stability using Lyapunov exponents.

This package provides tools to analyze how stable your neural network training is
by tracking weight trajectories and computing stability metrics from dynamical
systems theory.

Example:
    >>> from deep_lyapunov import StabilityAnalyzer
    >>> analyzer = StabilityAnalyzer(model, perturbation_scale=0.01)
    >>> results = analyzer.analyze(train_fn, train_loader, n_epochs=10)
    >>> print(f"Convergence: {results.convergence_ratio:.2f}x")
"""

from deep_lyapunov.analyzer import StabilityAnalyzer
from deep_lyapunov.config import AnalyzerConfig
from deep_lyapunov.results import AnalysisResults, TrajectoryMetrics

__version__ = "0.1.0"

__all__ = [
    "StabilityAnalyzer",
    "AnalyzerConfig",
    "AnalysisResults",
    "TrajectoryMetrics",
    "__version__",
]
