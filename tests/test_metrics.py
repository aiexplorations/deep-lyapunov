"""Tests for metrics module."""

import numpy as np
import pytest

from deep_lyapunov.metrics import (
    compute_convergence_ratio,
    compute_effective_dimensionality,
    compute_local_lyapunov,
    compute_lyapunov_exponent,
    compute_participation_ratio,
    compute_spread_evolution,
    project_to_pca,
)
from deep_lyapunov.metrics.convergence import (
    compute_manifold_center,
    compute_pairwise_distances,
    interpolate_trajectories,
)
from deep_lyapunov.metrics.dimensionality import (
    compute_cumulative_variance,
    compute_local_dimensionality,
    find_intrinsic_dimension,
)
from deep_lyapunov.metrics.lyapunov import compute_lyapunov_from_trajectories


class TestLyapunovExponent:
    """Tests for Lyapunov exponent computation."""

    def test_convergent_trajectory(self):
        """Test Lyapunov exponent for convergent spread."""
        # Spread decreases over time
        spread = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
        lyap = compute_lyapunov_exponent(spread, n_epochs=4)
        assert lyap < 0, "Convergent trajectory should have negative Lyapunov"

    def test_divergent_trajectory(self):
        """Test Lyapunov exponent for divergent spread."""
        # Spread increases over time
        spread = np.array([0.4, 0.5, 0.6, 0.8, 1.0])
        lyap = compute_lyapunov_exponent(spread, n_epochs=4)
        assert lyap > 0, "Divergent trajectory should have positive Lyapunov"

    def test_neutral_trajectory(self):
        """Test Lyapunov exponent for constant spread."""
        spread = np.array([1.0, 1.0, 1.0, 1.0])
        lyap = compute_lyapunov_exponent(spread, n_epochs=3)
        assert abs(lyap) < 0.01, "Constant spread should have Lyapunov near zero"

    def test_single_point(self):
        """Test with single spread value."""
        spread = np.array([1.0])
        lyap = compute_lyapunov_exponent(spread, n_epochs=1)
        assert lyap == 0.0

    def test_zero_spread(self):
        """Test with zero spread values."""
        spread = np.array([0.0, 0.0, 0.0])
        lyap = compute_lyapunov_exponent(spread, n_epochs=2)
        assert lyap == 0.0

    def test_zero_epochs(self):
        """Test with zero epochs uses default."""
        spread = np.array([1.0, 0.5])
        lyap = compute_lyapunov_exponent(spread, n_epochs=0)
        assert lyap < 0  # Should still compute correctly


class TestLocalLyapunov:
    """Tests for local Lyapunov exponent computation."""

    def test_basic_computation(self):
        """Test basic local Lyapunov computation."""
        spread = np.array([1.0, 0.8, 0.9, 0.7])
        local = compute_local_lyapunov(spread)
        assert len(local) == 3
        assert local[0] < 0  # 0.8/1.0 < 1
        assert local[1] > 0  # 0.9/0.8 > 1
        assert local[2] < 0  # 0.7/0.9 < 1

    def test_single_value(self):
        """Test with single spread value."""
        spread = np.array([1.0])
        local = compute_local_lyapunov(spread)
        assert len(local) == 1
        assert local[0] == 0.0

    def test_zero_values_handled(self):
        """Test that zero values are handled gracefully."""
        spread = np.array([0.0, 0.1, 0.2])
        local = compute_local_lyapunov(spread)
        # Should not produce inf/nan
        assert np.all(np.isfinite(local))


class TestLyapunovFromTrajectories:
    """Tests for trajectory-based Lyapunov computation."""

    def test_basic_computation(self, mock_trajectory):
        """Test basic Lyapunov from trajectories."""
        global_lyap, local_lyap = compute_lyapunov_from_trajectories(mock_trajectory)
        assert isinstance(global_lyap, float)
        assert isinstance(local_lyap, np.ndarray)
        assert len(local_lyap) == mock_trajectory.shape[0] - 1

    def test_convergent_trajectories(self, convergent_trajectory):
        """Test with convergent trajectory data."""
        global_lyap, _ = compute_lyapunov_from_trajectories(convergent_trajectory)
        assert global_lyap < 0

    def test_divergent_trajectories(self, divergent_trajectory):
        """Test with divergent trajectory data."""
        global_lyap, _ = compute_lyapunov_from_trajectories(divergent_trajectory)
        assert global_lyap > 0

    def test_single_trajectory(self):
        """Test with single trajectory."""
        traj = np.random.randn(10, 1, 50)
        global_lyap, local_lyap = compute_lyapunov_from_trajectories(traj)
        assert global_lyap == 0.0


class TestConvergenceRatio:
    """Tests for convergence ratio computation."""

    def test_convergent_ratio(self, convergent_trajectory):
        """Test ratio for convergent trajectories."""
        ratio = compute_convergence_ratio(convergent_trajectory)
        assert ratio < 1.0, "Convergent trajectories should have ratio < 1"

    def test_divergent_ratio(self, divergent_trajectory):
        """Test ratio for divergent trajectories."""
        ratio = compute_convergence_ratio(divergent_trajectory)
        assert ratio > 1.0, "Divergent trajectories should have ratio > 1"

    def test_single_trajectory(self):
        """Test with single trajectory."""
        traj = np.random.randn(10, 1, 50)
        ratio = compute_convergence_ratio(traj)
        assert ratio == 1.0

    def test_basic_computation(self, mock_trajectory):
        """Test basic ratio computation."""
        ratio = compute_convergence_ratio(mock_trajectory)
        assert isinstance(ratio, float)
        assert ratio > 0


class TestSpreadEvolution:
    """Tests for spread evolution computation."""

    def test_basic_computation(self, mock_trajectory):
        """Test basic spread evolution."""
        spread = compute_spread_evolution(mock_trajectory)
        assert len(spread) == mock_trajectory.shape[0]
        assert all(s >= 0 for s in spread)

    def test_convergent_spread_decreases(self, convergent_trajectory):
        """Test that convergent trajectories show decreasing spread."""
        spread = compute_spread_evolution(convergent_trajectory)
        # Overall trend should be decreasing
        assert spread[-1] < spread[0]

    def test_single_trajectory(self):
        """Test with single trajectory."""
        traj = np.random.randn(10, 1, 50)
        spread = compute_spread_evolution(traj)
        assert len(spread) == 10
        assert all(s == 0 for s in spread)


class TestPCAProjection:
    """Tests for PCA trajectory projection."""

    def test_basic_projection(self, mock_trajectory):
        """Test basic PCA projection."""
        n_components = 5
        projected, variance = project_to_pca(mock_trajectory, n_components)

        n_checkpoints, n_trajectories, _ = mock_trajectory.shape
        assert projected.shape[0] == n_checkpoints
        assert projected.shape[1] == n_trajectories
        assert projected.shape[2] == n_components
        assert len(variance) == n_components
        assert all(0 <= v <= 1 for v in variance)

    def test_variance_decreasing(self, mock_trajectory):
        """Test that explained variance is decreasing."""
        _, variance = project_to_pca(mock_trajectory, n_components=5)
        for i in range(len(variance) - 1):
            assert variance[i] >= variance[i + 1]

    def test_fewer_components_than_requested(self):
        """Test when fewer components available than requested."""
        # With 10*2=20 samples and 50 features, max components is min(20, 50)
        # But we request 10, so we should get 10
        traj = np.random.randn(10, 2, 50)
        projected, variance = project_to_pca(traj, n_components=10)
        assert projected.shape[2] == 10


class TestInterpolation:
    """Tests for trajectory interpolation."""

    def test_upsample(self, mock_trajectory):
        """Test upsampling trajectories."""
        target_length = 20
        interpolated = interpolate_trajectories(mock_trajectory, target_length)
        assert interpolated.shape[0] == target_length
        assert interpolated.shape[1:] == mock_trajectory.shape[1:]

    def test_downsample(self, mock_trajectory):
        """Test downsampling trajectories."""
        target_length = 5
        interpolated = interpolate_trajectories(mock_trajectory, target_length)
        assert interpolated.shape[0] == target_length

    def test_no_change(self, mock_trajectory):
        """Test with same length."""
        interpolated = interpolate_trajectories(
            mock_trajectory, mock_trajectory.shape[0]
        )
        np.testing.assert_array_equal(interpolated, mock_trajectory)

    def test_none_target(self, mock_trajectory):
        """Test with None target length."""
        interpolated = interpolate_trajectories(mock_trajectory, None)
        np.testing.assert_array_equal(interpolated, mock_trajectory)


class TestManifoldCenter:
    """Tests for manifold center computation."""

    def test_basic_computation(self, mock_trajectory):
        """Test basic manifold center."""
        center = compute_manifold_center(mock_trajectory)
        n_checkpoints, _, n_params = mock_trajectory.shape
        assert center.shape == (n_checkpoints, n_params)

    def test_center_is_mean(self, mock_trajectory):
        """Test that center is the mean across trajectories."""
        center = compute_manifold_center(mock_trajectory)
        expected = np.mean(mock_trajectory, axis=1)
        np.testing.assert_array_almost_equal(center, expected)


class TestPairwiseDistances:
    """Tests for pairwise distance computation."""

    def test_basic_computation(self):
        """Test basic pairwise distances."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        distances = compute_pairwise_distances(points)
        assert distances.shape == (3, 3)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(distances), [0, 0, 0])
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_single_point(self):
        """Test with single point."""
        points = np.array([[1, 2, 3]])
        distances = compute_pairwise_distances(points)
        assert distances.shape == (1, 1)
        assert distances[0, 0] == 0.0


class TestEffectiveDimensionality:
    """Tests for effective dimensionality computation."""

    def test_basic_computation(self, mock_trajectory):
        """Test basic effective dimensionality."""
        eff_dim = compute_effective_dimensionality(mock_trajectory)
        assert len(eff_dim) > 0
        assert all(d >= 1 for d in eff_dim)

    def test_small_trajectory(self):
        """Test with small trajectory."""
        traj = np.random.randn(5, 3, 20)
        eff_dim = compute_effective_dimensionality(traj)
        assert len(eff_dim) >= 1


class TestParticipationRatio:
    """Tests for participation ratio computation."""

    def test_one_dimensional_data(self):
        """Test PR for 1D data."""
        # All variance in one dimension
        data = np.column_stack([np.random.randn(100), np.zeros(100)])
        pr = compute_participation_ratio(data)
        assert pr < 2, "1D data should have PR close to 1"

    def test_uniform_variance(self):
        """Test PR for data with uniform variance."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        pr = compute_participation_ratio(data)
        # Should be close to number of dimensions
        assert pr > 3, "Uniform variance should have high PR"

    def test_single_sample(self):
        """Test with single sample."""
        data = np.array([[1, 2, 3]])
        pr = compute_participation_ratio(data)
        assert pr == 1.0

    def test_empty_data(self):
        """Test with minimal data."""
        data = np.array([[1]])
        pr = compute_participation_ratio(data)
        assert pr == 1.0


class TestCumulativeVariance:
    """Tests for cumulative variance computation."""

    def test_basic_computation(self):
        """Test basic cumulative variance."""
        var_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        cum_var = compute_cumulative_variance(var_ratio)
        np.testing.assert_array_almost_equal(cum_var, [0.5, 0.8, 0.95, 1.0])


class TestIntrinsicDimension:
    """Tests for intrinsic dimension finding."""

    def test_basic_computation(self):
        """Test finding intrinsic dimension."""
        var_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        n_dims = find_intrinsic_dimension(var_ratio, threshold=0.95)
        assert n_dims == 3

    def test_threshold_one(self):
        """Test with threshold of 1.0."""
        var_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        n_dims = find_intrinsic_dimension(var_ratio, threshold=1.0)
        assert n_dims == 4

    def test_threshold_low(self):
        """Test with low threshold."""
        var_ratio = np.array([0.5, 0.3, 0.15, 0.05])
        n_dims = find_intrinsic_dimension(var_ratio, threshold=0.5)
        assert n_dims == 1


class TestLocalDimensionality:
    """Tests for local dimensionality computation."""

    def test_basic_computation(self):
        """Test basic local dimensionality."""
        trajectory = np.random.randn(20, 50)
        local_dim = compute_local_dimensionality(trajectory, k_neighbors=5)
        assert len(local_dim) == 20
        assert all(d >= 1 for d in local_dim)

    def test_short_trajectory(self):
        """Test with short trajectory."""
        trajectory = np.random.randn(3, 50)
        local_dim = compute_local_dimensionality(trajectory, k_neighbors=10)
        assert len(local_dim) == 3

    def test_very_short_trajectory_fallback(self):
        """Test local_window < 3 returns 1.0 (line 207)."""
        # With only 2 points and k_neighbors=2, local_window will be < 3
        trajectory = np.random.randn(2, 50)
        local_dim = compute_local_dimensionality(trajectory, k_neighbors=2)
        assert len(local_dim) == 2
        # Both should be 1.0 due to fallback


class TestParticipationRatioEdgeCases:
    """Tests for edge cases in participation ratio computation."""

    def test_single_feature_scalar_cov(self):
        """Test participation ratio with single feature (scalar covariance, line 108)."""
        # Single feature produces scalar covariance
        data = np.random.randn(10, 1)
        pr = compute_participation_ratio(data)
        assert pr == 1.0

    def test_zero_variance_data(self):
        """Test participation ratio with zero variance (sum_eig_sq <= 0, line 124)."""
        # All same values - zero variance
        data = np.ones((10, 5))
        pr = compute_participation_ratio(data)
        assert pr == 1.0

    def test_numerical_edge_case(self):
        """Test participation ratio with near-zero eigenvalues."""
        # Data with very small variance
        data = np.ones((10, 3)) + np.random.randn(10, 3) * 1e-15
        pr = compute_participation_ratio(data)
        assert pr >= 1.0


class TestEffectiveDimensionalityEdgeCases:
    """Tests for edge cases in effective dimensionality."""

    def test_large_window_fallback(self):
        """Test fallback when window_size > n_checkpoints."""
        # Very small trajectory where window would be too large
        traj = np.random.randn(5, 3, 20)
        eff_dim = compute_effective_dimensionality(traj, window_size=100)
        # Should fall back to global computation
        assert len(eff_dim) >= 1


class TestConvergenceRatioEdgeCases:
    """Tests for convergence ratio edge cases."""

    def test_divergent_trajectories_large_ratio(self):
        """Test convergence ratio for strongly divergent trajectories."""
        # Create trajectories that start close and diverge significantly
        n_checkpoints, n_trajectories, n_params = 10, 5, 20
        np.random.seed(42)
        traj = np.zeros((n_checkpoints, n_trajectories, n_params))

        # All start with small differences
        for i in range(n_trajectories):
            traj[0, i, :] = np.random.randn(n_params) * 0.001

        # Diverge over time with exponential growth
        for t in range(1, n_checkpoints):
            for i in range(n_trajectories):
                traj[t, i, :] = traj[t - 1, i, :] + np.random.randn(n_params) * (
                    t * 0.5
                )

        ratio = compute_convergence_ratio(traj)
        # Should have ratio > 1 for divergent trajectories
        assert ratio > 1.0

    def test_convergent_trajectories_small_ratio(self):
        """Test convergence ratio for convergent trajectories."""
        # Create trajectories that start apart and converge
        n_checkpoints, n_trajectories, n_params = 10, 5, 20
        np.random.seed(42)
        traj = np.zeros((n_checkpoints, n_trajectories, n_params))

        # Start with large differences
        target = np.random.randn(n_params)
        for i in range(n_trajectories):
            traj[0, i, :] = target + np.random.randn(n_params) * 2.0

        # Converge to target over time
        for t in range(1, n_checkpoints):
            alpha = t / n_checkpoints  # Interpolation factor
            for i in range(n_trajectories):
                traj[t, i, :] = (1 - alpha) * traj[0, i, :] + alpha * target

        ratio = compute_convergence_ratio(traj)
        # Should have ratio < 1 for convergent trajectories
        assert ratio < 1.0
