"""Tests for results module."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from deep_lyapunov.results import AnalysisResults, TrajectoryMetrics


class TestTrajectoryMetrics:
    """Tests for TrajectoryMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        tm = TrajectoryMetrics(trajectory_id=0)
        assert tm.trajectory_id == 0
        assert tm.final_loss is None
        assert tm.final_accuracy is None
        assert tm.path_length == 0.0

    def test_with_values(self):
        """Test creation with all values."""
        tm = TrajectoryMetrics(
            trajectory_id=1,
            final_loss=0.5,
            final_accuracy=0.95,
            path_length=10.5,
            velocity_mean=1.2,
            velocity_std=0.3,
        )
        assert tm.trajectory_id == 1
        assert tm.final_loss == 0.5
        assert tm.final_accuracy == 0.95
        assert tm.path_length == 10.5


class TestAnalysisResults:
    """Tests for AnalysisResults dataclass."""

    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results."""
        return AnalysisResults(
            convergence_ratio=0.75,
            lyapunov=-0.03,
            behavior="convergent",
            spread_evolution=np.array([1.0, 0.9, 0.8, 0.75]),
            pca_trajectories=np.random.randn(4, 3, 5),
            pca_explained_variance=np.array([0.5, 0.3, 0.15, 0.04, 0.01]),
            effective_dimensionality=3.2,
            trajectory_metrics=[
                TrajectoryMetrics(trajectory_id=0, final_loss=0.1),
                TrajectoryMetrics(trajectory_id=1, final_loss=0.12),
                TrajectoryMetrics(trajectory_id=2, final_loss=0.11),
            ],
            n_trajectories=3,
            n_checkpoints=4,
            n_parameters=1000,
        )

    def test_creation(self, sample_results):
        """Test basic creation."""
        assert sample_results.convergence_ratio == 0.75
        assert sample_results.lyapunov == -0.03
        assert sample_results.behavior == "convergent"
        assert sample_results.n_trajectories == 3

    def test_to_dict(self, sample_results):
        """Test conversion to dictionary."""
        result_dict = sample_results.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["convergence_ratio"] == 0.75
        assert result_dict["lyapunov"] == -0.03
        assert result_dict["behavior"] == "convergent"
        assert result_dict["n_trajectories"] == 3
        assert "spread_evolution" in result_dict
        assert "trajectory_metrics" in result_dict
        assert len(result_dict["trajectory_metrics"]) == 3

    def test_to_json(self, sample_results):
        """Test JSON serialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            sample_results.to_json(path)
            with open(path) as f:
                data = json.load(f)

            assert data["convergence_ratio"] == 0.75
            assert data["behavior"] == "convergent"
        finally:
            os.unlink(path)

    def test_from_dict(self, sample_results):
        """Test reconstruction from dictionary."""
        result_dict = sample_results.to_dict()
        reconstructed = AnalysisResults.from_dict(result_dict)

        assert reconstructed.convergence_ratio == sample_results.convergence_ratio
        assert reconstructed.lyapunov == sample_results.lyapunov
        assert reconstructed.behavior == sample_results.behavior
        assert reconstructed.n_trajectories == sample_results.n_trajectories
        assert len(reconstructed.trajectory_metrics) == len(
            sample_results.trajectory_metrics
        )

    def test_plot_trajectories(self, sample_results):
        """Test trajectory plotting."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig = sample_results.plot_trajectories()
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectories_custom_components(self, sample_results):
        """Test trajectory plotting with custom components."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = sample_results.plot_trajectories(components=(0, 1))
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence(self, sample_results):
        """Test convergence plotting."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = sample_results.plot_convergence()
        assert fig is not None
        plt.close(fig)

    def test_plot_lyapunov(self, sample_results):
        """Test Lyapunov plotting."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = sample_results.plot_lyapunov()
        assert fig is not None
        plt.close(fig)

    def test_save_report_html_embedded(self, sample_results):
        """Test HTML report generation with embedded images (default)."""
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_results.save_report(tmpdir)

            # Check files were created - embedded means no PNG files
            assert (Path(tmpdir) / "metrics.json").exists()
            assert (Path(tmpdir) / "report.html").exists()

            # PNG files should NOT exist when embedded
            assert not (Path(tmpdir) / "trajectories.png").exists()
            assert not (Path(tmpdir) / "convergence.png").exists()
            assert not (Path(tmpdir) / "lyapunov.png").exists()

            # Check HTML report contains embedded images
            with open(Path(tmpdir) / "report.html") as f:
                report = f.read()
            assert "Stability Analysis Report" in report
            assert "data:image/png;base64," in report  # Embedded images
            assert "0.75" in report  # convergence ratio

    def test_save_report_separate_images(self, sample_results):
        """Test report generation with separate image files."""
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_results.save_report(tmpdir, embed_images=False)

            # Check files were created
            assert (Path(tmpdir) / "trajectories.png").exists()
            assert (Path(tmpdir) / "convergence.png").exists()
            assert (Path(tmpdir) / "lyapunov.png").exists()
            assert (Path(tmpdir) / "metrics.json").exists()
            assert (Path(tmpdir) / "report.html").exists()

    def test_save_report_markdown_format(self, sample_results):
        """Test markdown format report generation."""
        import matplotlib

        matplotlib.use("Agg")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_results.save_report(tmpdir, format="markdown")

            # Check markdown file was created
            assert (Path(tmpdir) / "report.md").exists()
            assert not (Path(tmpdir) / "report.html").exists()

            # Check markdown report content
            with open(Path(tmpdir) / "report.md") as f:
                report = f.read()
            assert "# Stability Analysis Report" in report

    def test_save_report_invalid_path(self, sample_results):
        """Test that file-like paths are rejected."""
        import pytest

        with pytest.raises(ValueError, match="looks like a file path"):
            sample_results.save_report("my_report.json")

    def test_generate_markdown_report(self, sample_results):
        """Test markdown report generation."""
        report = sample_results._generate_markdown_report()

        assert "# Stability Analysis Report" in report
        assert "Convergence Ratio" in report
        assert "Lyapunov Exponent" in report
        assert "convergent" in report.lower()


class TestAnalysisResultsDivergent:
    """Tests for divergent analysis results."""

    @pytest.fixture
    def divergent_results(self):
        """Create divergent analysis results."""
        return AnalysisResults(
            convergence_ratio=1.5,
            lyapunov=0.04,
            behavior="divergent",
            spread_evolution=np.array([1.0, 1.1, 1.3, 1.5]),
            pca_trajectories=np.random.randn(4, 3, 5),
            pca_explained_variance=np.array([0.5, 0.3, 0.15, 0.04, 0.01]),
            effective_dimensionality=4.1,
            n_trajectories=3,
            n_checkpoints=4,
            n_parameters=1000,
        )

    def test_divergent_behavior(self, divergent_results):
        """Test divergent behavior labeling."""
        assert divergent_results.behavior == "divergent"
        assert divergent_results.convergence_ratio > 1.0
        assert divergent_results.lyapunov > 0

    def test_divergent_report(self, divergent_results):
        """Test markdown report for divergent results."""
        report = divergent_results._generate_markdown_report()
        assert "divergent" in report.lower()


class TestAnalysisResultsPlotting:
    """Additional plotting tests."""

    @pytest.fixture
    def minimal_results(self):
        """Create minimal results for edge case testing."""
        return AnalysisResults(
            convergence_ratio=1.0,
            lyapunov=0.0,
            behavior="convergent",
            spread_evolution=np.array([1.0, 1.0]),
            pca_trajectories=np.random.randn(2, 2, 3),
            pca_explained_variance=np.array([0.7, 0.2, 0.1]),
            effective_dimensionality=2.0,
            n_trajectories=2,
            n_checkpoints=2,
            n_parameters=100,
        )

    def test_plot_with_existing_axes(self, minimal_results):
        """Test plotting on existing axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = minimal_results.plot_trajectories(ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_plot_convergence_with_axes(self, minimal_results):
        """Test convergence plot on existing axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = minimal_results.plot_convergence(ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_plot_lyapunov_with_axes(self, minimal_results):
        """Test Lyapunov plot on existing axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = minimal_results.plot_lyapunov(ax=ax)
        assert result_fig is fig
        plt.close(fig)

    def test_plot_lyapunov_minimal_data(self):
        """Test Lyapunov plot with minimal data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        results = AnalysisResults(
            convergence_ratio=1.0,
            lyapunov=0.0,
            behavior="convergent",
            spread_evolution=np.array([1.0]),  # Only one point
            pca_trajectories=np.random.randn(1, 2, 3),
            pca_explained_variance=np.array([0.8, 0.15, 0.05]),
            effective_dimensionality=1.5,
            n_trajectories=2,
            n_checkpoints=1,
            n_parameters=100,
        )

        fig = results.plot_lyapunov()
        assert fig is not None
        plt.close(fig)
