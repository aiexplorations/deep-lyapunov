"""Tests for tracking module."""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from deep_lyapunov.tracking import (
    TrajectoryTracker,
    apply_perturbation,
    create_perturbed_copies,
)
from deep_lyapunov.tracking.perturbation import (
    compute_perturbation_magnitude,
    get_parameter_statistics,
)


class TestTrajectoryTracker:
    """Tests for TrajectoryTracker class."""

    def test_init(self, simple_model):
        """Test tracker initialization."""
        tracker = TrajectoryTracker(simple_model)

        assert tracker.model is simple_model
        assert tracker.n_snapshots == 0
        assert tracker.n_parameters > 0

    def test_capture_weights(self, simple_model):
        """Test weight capture."""
        tracker = TrajectoryTracker(simple_model)

        tracker.capture()
        assert tracker.n_snapshots == 1

        tracker.capture()
        assert tracker.n_snapshots == 2

    def test_get_trajectory(self, simple_model):
        """Test trajectory retrieval."""
        tracker = TrajectoryTracker(simple_model)

        # Capture multiple checkpoints
        for _ in range(5):
            tracker.capture()

        trajectory = tracker.get_trajectory()

        assert trajectory.shape[0] == 5
        assert trajectory.shape[1] == tracker.n_parameters

    def test_get_trajectory_empty(self, simple_model):
        """Test trajectory retrieval with no captures."""
        tracker = TrajectoryTracker(simple_model)

        with pytest.raises(ValueError, match="No snapshots captured"):
            tracker.get_trajectory()

    def test_record_interval(self, simple_model):
        """Test record interval functionality."""
        tracker = TrajectoryTracker(simple_model, record_interval=2)

        # Capture 5 times, but only every 2nd should be recorded
        for _ in range(5):
            tracker.capture()

        # Should have 2 snapshots (captures 2 and 4)
        assert tracker.n_snapshots == 2

    def test_get_latest_weights(self, simple_model):
        """Test getting latest weights."""
        tracker = TrajectoryTracker(simple_model)

        assert tracker.get_latest_weights() is None

        tracker.capture()
        latest = tracker.get_latest_weights()

        assert latest is not None
        assert len(latest) == tracker.n_parameters

    def test_reset(self, simple_model):
        """Test tracker reset."""
        tracker = TrajectoryTracker(simple_model)

        tracker.capture()
        tracker.capture()
        assert tracker.n_snapshots == 2

        tracker.reset()
        assert tracker.n_snapshots == 0

    def test_get_statistics(self, simple_model):
        """Test trajectory statistics computation."""
        tracker = TrajectoryTracker(simple_model)

        # Need at least 2 snapshots for meaningful stats
        tracker.capture()

        # Modify weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        tracker.capture()

        stats = tracker.get_statistics()

        assert "path_length" in stats
        assert "velocity_mean" in stats
        assert "velocity_std" in stats
        assert stats["path_length"] > 0

    def test_track_gradients(self, simple_model):
        """Test gradient tracking."""
        tracker = TrajectoryTracker(simple_model, track_gradients=True)

        # Create dummy gradients
        x = torch.randn(4, 4)
        y = simple_model(x)
        y.sum().backward()

        tracker.capture()

        grad_traj = tracker.get_gradient_trajectory()
        assert grad_traj is not None
        assert grad_traj.shape[1] == tracker.n_parameters


class TestApplyPerturbation:
    """Tests for apply_perturbation function."""

    def test_perturbation_modifies_weights(self, simple_model):
        """Test that perturbation actually modifies weights."""
        original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        apply_perturbation(simple_model, scale=0.1, seed=42)

        # Check that at least one parameter changed
        changed = False
        for name, param in simple_model.named_parameters():
            if not torch.allclose(param, original_state[name]):
                changed = True
                break

        assert changed, "Perturbation should modify weights"

    def test_perturbation_reproducibility(self, simple_model):
        """Test that same seed produces same perturbation."""
        model1 = copy.deepcopy(simple_model)
        model2 = copy.deepcopy(simple_model)

        apply_perturbation(model1, scale=0.1, seed=42)
        apply_perturbation(model2, scale=0.1, seed=42)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_perturbation_scale(self, simple_model):
        """Test that perturbation magnitude scales with scale parameter."""
        model_small = copy.deepcopy(simple_model)
        model_large = copy.deepcopy(simple_model)
        original = copy.deepcopy(simple_model)

        apply_perturbation(model_small, scale=0.01, seed=42)
        apply_perturbation(model_large, scale=0.1, seed=42)

        dist_small = compute_perturbation_magnitude(original, model_small)
        dist_large = compute_perturbation_magnitude(original, model_large)

        # Larger scale should produce larger perturbation
        assert dist_large > dist_small


class TestCreatePerturbedCopies:
    """Tests for create_perturbed_copies function."""

    def test_creates_correct_number(self, simple_model):
        """Test that correct number of copies is created."""
        copies = create_perturbed_copies(simple_model, n_copies=5)
        assert len(copies) == 5

    def test_first_copy_unperturbed(self, simple_model):
        """Test that first copy is unperturbed when include_original=True."""
        copies = create_perturbed_copies(
            simple_model, n_copies=3, include_original=True, base_seed=42
        )

        # First should match original
        for p_orig, p_copy in zip(simple_model.parameters(), copies[0].parameters()):
            assert torch.allclose(p_orig, p_copy)

    def test_copies_are_different(self, simple_model):
        """Test that different copies have different weights."""
        copies = create_perturbed_copies(simple_model, n_copies=3, base_seed=42)

        # Compare copy 1 and copy 2
        different = False
        for p1, p2 in zip(copies[1].parameters(), copies[2].parameters()):
            if not torch.allclose(p1, p2):
                different = True
                break

        assert different, "Different copies should have different weights"

    def test_all_perturbed_when_not_include_original(self, simple_model):
        """Test that all copies are perturbed when include_original=False."""
        copies = create_perturbed_copies(
            simple_model, n_copies=3, include_original=False, base_seed=42
        )

        # All should be different from original
        for i, copy in enumerate(copies):
            different = False
            for p_orig, p_copy in zip(simple_model.parameters(), copy.parameters()):
                if not torch.allclose(p_orig, p_copy):
                    different = True
                    break
            assert different, f"Copy {i} should be perturbed"


class TestPerturbationUtilities:
    """Tests for perturbation utility functions."""

    def test_compute_perturbation_magnitude(self, simple_model):
        """Test perturbation magnitude computation."""
        model_copy = copy.deepcopy(simple_model)
        apply_perturbation(model_copy, scale=0.1, seed=42)

        magnitude = compute_perturbation_magnitude(simple_model, model_copy)

        assert magnitude > 0
        assert isinstance(magnitude, float)

    def test_get_parameter_statistics(self, simple_model):
        """Test parameter statistics computation."""
        stats = get_parameter_statistics(simple_model)

        assert "n_parameters" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["n_parameters"] > 0
        assert stats["std"] > 0  # Initialized weights have variance

    def test_get_parameter_statistics_empty_model(self):
        """Test parameter statistics for model with no trainable params."""

        class NoParamModel(nn.Module):
            def forward(self, x):
                return x

        model = NoParamModel()
        stats = get_parameter_statistics(model)

        assert stats["n_parameters"] == 0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0


class TestTrackerEdgeCases:
    """Tests for tracker edge cases."""

    def test_get_statistics_single_snapshot(self, simple_model):
        """Test statistics with only 1 snapshot (line 183)."""
        tracker = TrajectoryTracker(simple_model)
        tracker.capture()

        stats = tracker.get_statistics()

        # Should return zeros when < 2 snapshots
        assert stats["path_length"] == 0.0
        assert stats["velocity_mean"] == 0.0
        assert stats["velocity_std"] == 0.0

    def test_get_gradient_trajectory_none(self, simple_model):
        """Test gradient trajectory returns None when not tracked (line 141)."""
        tracker = TrajectoryTracker(simple_model, track_gradients=False)
        tracker.capture()

        grad_traj = tracker.get_gradient_trajectory()
        assert grad_traj is None

    def test_gradient_zeros_when_none(self, simple_model):
        """Test gradient capture fills zeros when grad is None (line 94)."""
        tracker = TrajectoryTracker(simple_model, track_gradients=True)

        # Capture without running backward - no gradients computed
        tracker.capture()

        # Should have captured gradient trajectory (with zeros)
        grad_traj = tracker.get_gradient_trajectory()
        # If no gradients at all (all None), returns None
        # But if some params have grads, fills zeros for those that don't
        # In this case, no backward called, so all None -> returns None
        assert grad_traj is None

    def test_gradient_partial_zeros(self, simple_model):
        """Test gradient capture with partial gradients."""
        tracker = TrajectoryTracker(simple_model, track_gradients=True)

        # Create a situation where some params have gradients
        x = torch.randn(4, 4)
        y = simple_model(x)
        y.sum().backward()

        # Now clear only some gradients
        for i, param in enumerate(simple_model.parameters()):
            if i == 0:
                param.grad = None  # Clear first param's grad

        tracker.capture()

        # Should have captured with zeros for the cleared gradient
        grad_traj = tracker.get_gradient_trajectory()
        assert grad_traj is not None


class TestPerturbationZeroStd:
    """Tests for perturbation with zero std parameters."""

    def test_perturbation_zero_std_param(self):
        """Test perturbation when parameter has zero std."""

        class ConstantParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(10))  # All same value -> std=0

            def forward(self, x):
                return x * self.param

        model = ConstantParamModel()
        original = model.param.clone()

        apply_perturbation(model, scale=0.01, seed=42)

        # Should still apply perturbation using default std of 0.01
        assert not torch.allclose(model.param, original)
