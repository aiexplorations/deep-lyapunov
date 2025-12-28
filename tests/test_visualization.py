"""Tests for visualization module."""

import tempfile

import numpy as np
import pytest


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Set up matplotlib for testing."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

    @pytest.fixture
    def sample_data(self):
        """Create sample visualization data."""
        np.random.seed(42)
        return {
            "pca_trajectories": np.random.randn(10, 5, 5),
            "pca_variance": np.array([0.4, 0.25, 0.15, 0.12, 0.08]),
            "spread_evolution": np.array(
                [1.0, 0.9, 0.85, 0.8, 0.78, 0.75, 0.73, 0.71, 0.7, 0.68]
            ),
            "lyapunov": -0.035,
            "convergence_ratio": 0.68,
            "behavior": "convergent",
        }

    def test_import_visualization(self):
        """Test that visualization module can be imported."""
        from deep_lyapunov.visualization import plots

        assert hasattr(plots, "plot_trajectories_2d")
        assert hasattr(plots, "plot_trajectories_3d")
        assert hasattr(plots, "plot_spread_evolution")
        assert hasattr(plots, "plot_lyapunov_evolution")
        assert hasattr(plots, "create_analysis_dashboard")

    def test_plot_trajectories_2d(self, sample_data):
        """Test 2D trajectory plotting."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_2d

        fig = plot_trajectories_2d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectories_2d_custom_components(self, sample_data):
        """Test 2D plotting with custom components."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_2d

        fig = plot_trajectories_2d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            components=(1, 2),
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectories_2d_no_endpoints(self, sample_data):
        """Test 2D plotting without endpoint markers."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_2d

        fig = plot_trajectories_2d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            show_endpoints=False,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectories_2d_with_axes(self, sample_data):
        """Test 2D plotting on existing axes."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_2d

        fig, ax = plt.subplots()
        result = plot_trajectories_2d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            ax=ax,
        )
        assert result is fig
        plt.close(fig)

    def test_plot_trajectories_3d(self, sample_data):
        """Test 3D trajectory plotting."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_3d

        fig = plot_trajectories_3d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_trajectories_3d_custom_components(self, sample_data):
        """Test 3D plotting with custom components."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_3d

        fig = plot_trajectories_3d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            components=(0, 2, 4),
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_spread_evolution_convergent(self, sample_data):
        """Test spread evolution plot for convergent behavior."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_spread_evolution

        fig = plot_spread_evolution(
            sample_data["spread_evolution"],
            behavior="convergent",
            convergence_ratio=0.68,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_spread_evolution_divergent(self):
        """Test spread evolution plot for divergent behavior."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_spread_evolution

        spread = np.array([0.5, 0.6, 0.7, 0.85, 1.0])
        fig = plot_spread_evolution(
            spread,
            behavior="divergent",
            convergence_ratio=2.0,
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_spread_evolution_with_axes(self, sample_data):
        """Test spread evolution on existing axes."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_spread_evolution

        fig, ax = plt.subplots()
        result = plot_spread_evolution(
            sample_data["spread_evolution"],
            ax=ax,
        )
        assert result is fig
        plt.close(fig)

    def test_plot_lyapunov_evolution(self, sample_data):
        """Test Lyapunov evolution plot."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_lyapunov_evolution

        fig = plot_lyapunov_evolution(
            sample_data["spread_evolution"],
            sample_data["lyapunov"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_lyapunov_evolution_short_data(self):
        """Test Lyapunov plot with minimal data."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_lyapunov_evolution

        spread = np.array([1.0])  # Single point
        fig = plot_lyapunov_evolution(spread, 0.0)
        assert fig is not None
        plt.close(fig)

    def test_plot_lyapunov_evolution_with_axes(self, sample_data):
        """Test Lyapunov evolution on existing axes."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_lyapunov_evolution

        fig, ax = plt.subplots()
        result = plot_lyapunov_evolution(
            sample_data["spread_evolution"],
            sample_data["lyapunov"],
            ax=ax,
        )
        assert result is fig
        plt.close(fig)

    def test_plot_convergence_basin(self, sample_data):
        """Test convergence basin plot."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_convergence_basin

        fig = plot_convergence_basin(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_basin_custom_components(self, sample_data):
        """Test convergence basin with custom components."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_convergence_basin

        fig = plot_convergence_basin(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            components=(1, 2),
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_basin_few_points(self):
        """Test convergence basin with too few points for convex hull."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_convergence_basin

        # Only 2 trajectories - can't form convex hull
        pca_traj = np.random.randn(10, 2, 5)
        pca_var = np.array([0.5, 0.3, 0.15, 0.04, 0.01])

        fig = plot_convergence_basin(pca_traj, pca_var)
        assert fig is not None
        plt.close(fig)

    def test_create_analysis_dashboard(self, sample_data):
        """Test dashboard creation."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import create_analysis_dashboard

        fig = create_analysis_dashboard(
            pca_trajectories=sample_data["pca_trajectories"],
            pca_variance=sample_data["pca_variance"],
            spread_evolution=sample_data["spread_evolution"],
            lyapunov=sample_data["lyapunov"],
            convergence_ratio=sample_data["convergence_ratio"],
            behavior=sample_data["behavior"],
        )
        assert fig is not None
        plt.close(fig)

    def test_create_analysis_dashboard_save(self, sample_data):
        """Test dashboard saving."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import create_analysis_dashboard

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        fig = create_analysis_dashboard(
            pca_trajectories=sample_data["pca_trajectories"],
            pca_variance=sample_data["pca_variance"],
            spread_evolution=sample_data["spread_evolution"],
            lyapunov=sample_data["lyapunov"],
            convergence_ratio=sample_data["convergence_ratio"],
            behavior=sample_data["behavior"],
            save_path=path,
        )
        assert fig is not None
        plt.close(fig)

        import os

        assert os.path.exists(path)
        os.unlink(path)

    def test_colormap_option(self, sample_data):
        """Test custom colormap."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_2d

        fig = plot_trajectories_2d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            colormap="plasma",
        )
        assert fig is not None
        plt.close(fig)

    def test_alpha_option(self, sample_data):
        """Test custom alpha transparency."""
        import matplotlib.pyplot as plt

        from deep_lyapunov.visualization.plots import plot_trajectories_2d

        fig = plot_trajectories_2d(
            sample_data["pca_trajectories"],
            sample_data["pca_variance"],
            alpha=0.5,
        )
        assert fig is not None
        plt.close(fig)


class TestVisualizationInit:
    """Tests for visualization module __init__."""

    def test_init_imports(self):
        """Test that __init__ exports key functions."""
        from deep_lyapunov.visualization import (
            create_analysis_dashboard,
            plot_convergence_basin,
            plot_lyapunov_evolution,
            plot_spread_evolution,
            plot_trajectories_2d,
            plot_trajectories_3d,
        )

        # All should be callable
        assert callable(plot_trajectories_2d)
        assert callable(plot_trajectories_3d)
        assert callable(plot_spread_evolution)
        assert callable(plot_lyapunov_evolution)
        assert callable(plot_convergence_basin)
        assert callable(create_analysis_dashboard)
