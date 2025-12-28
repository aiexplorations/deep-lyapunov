"""Tests for configuration dataclasses."""

import pytest

from deep_lyapunov.config import AnalyzerConfig, MetricsConfig, VisualizationConfig


class TestAnalyzerConfig:
    """Tests for AnalyzerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AnalyzerConfig()

        assert config.perturbation_scale == 0.01
        assert config.n_trajectories == 5
        assert config.n_pca_components == 10
        assert config.track_gradients is False
        assert config.record_interval == 1
        assert config.device == "auto"
        assert config.seed == 42
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AnalyzerConfig(
            perturbation_scale=0.05,
            n_trajectories=10,
            n_pca_components=20,
            track_gradients=True,
            device="cpu",
            seed=123,
        )

        assert config.perturbation_scale == 0.05
        assert config.n_trajectories == 10
        assert config.n_pca_components == 20
        assert config.track_gradients is True
        assert config.device == "cpu"
        assert config.seed == 123

    def test_validation_perturbation_scale(self):
        """Test validation of perturbation_scale."""
        with pytest.raises(ValueError, match="perturbation_scale must be positive"):
            AnalyzerConfig(perturbation_scale=0)

        with pytest.raises(ValueError, match="perturbation_scale must be positive"):
            AnalyzerConfig(perturbation_scale=-0.1)

    def test_validation_n_trajectories(self):
        """Test validation of n_trajectories."""
        with pytest.raises(ValueError, match="n_trajectories must be at least 2"):
            AnalyzerConfig(n_trajectories=1)

    def test_validation_n_pca_components(self):
        """Test validation of n_pca_components."""
        with pytest.raises(ValueError, match="n_pca_components must be at least 1"):
            AnalyzerConfig(n_pca_components=0)

    def test_validation_record_interval(self):
        """Test validation of record_interval."""
        with pytest.raises(ValueError, match="record_interval must be at least 1"):
            AnalyzerConfig(record_interval=0)


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_values(self):
        """Test default metrics configuration."""
        config = MetricsConfig()

        assert config.convergence_threshold == 1.0
        assert config.lyapunov_window == 5
        assert config.min_trajectory_length == 10
        assert config.use_robust_estimators is True

    def test_custom_values(self):
        """Test custom metrics configuration."""
        config = MetricsConfig(
            convergence_threshold=0.5,
            lyapunov_window=10,
            min_trajectory_length=20,
            use_robust_estimators=False,
        )

        assert config.convergence_threshold == 0.5
        assert config.lyapunov_window == 10
        assert config.min_trajectory_length == 20
        assert config.use_robust_estimators is False


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_values(self):
        """Test default visualization configuration."""
        config = VisualizationConfig()

        assert config.figure_dpi == 150
        assert config.figure_format == "png"
        assert config.colormap == "viridis"
        assert config.show_confidence_bands is True

    def test_custom_values(self):
        """Test custom visualization configuration."""
        config = VisualizationConfig(
            figure_dpi=300,
            figure_format="pdf",
            colormap="plasma",
            show_confidence_bands=False,
        )

        assert config.figure_dpi == 300
        assert config.figure_format == "pdf"
        assert config.colormap == "plasma"
        assert config.show_confidence_bands is False
