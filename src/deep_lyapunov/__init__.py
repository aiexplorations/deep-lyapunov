"""deep-lyapunov: Analyze neural network training stability using Lyapunov exponents.

This package provides tools to analyze how stable your neural network training is
by tracking weight trajectories and computing stability metrics from dynamical
systems theory.

Example:
    >>> from deep_lyapunov import StabilityAnalyzer
    >>> analyzer = StabilityAnalyzer(model, perturbation_scale=0.01)
    >>> results = analyzer.analyze(train_fn, train_loader, n_epochs=10)
    >>> print(f"Convergence: {results.convergence_ratio:.2f}x")

Logging:
    The package uses Python's logging module. To see detailed output:

    >>> import logging
    >>> logging.basicConfig(level=logging.INFO)

    For debug-level details:

    >>> logging.getLogger("deep_lyapunov").setLevel(logging.DEBUG)
"""

import logging

from deep_lyapunov.analyzer import StabilityAnalyzer
from deep_lyapunov.config import AnalyzerConfig
from deep_lyapunov.results import AnalysisResults, TrajectoryMetrics

__version__ = "0.1.2"

__all__ = [
    "StabilityAnalyzer",
    "AnalyzerConfig",
    "AnalysisResults",
    "TrajectoryMetrics",
    "__version__",
    "get_logger",
]

# Package-level logger
logger = logging.getLogger(__name__)


def get_logger() -> logging.Logger:
    """Get the deep_lyapunov logger for configuration.

    Returns:
        The package logger instance.

    Example:
        >>> from deep_lyapunov import get_logger
        >>> logger = get_logger()
        >>> logger.setLevel(logging.DEBUG)
    """
    return logger


# Add a NullHandler to prevent "No handler found" warnings
# Users can add their own handlers as needed
logger.addHandler(logging.NullHandler())
