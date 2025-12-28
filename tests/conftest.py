"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_model():
    """Create a simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


@pytest.fixture
def tiny_model():
    """Create a tiny model for fast tests."""
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )


@pytest.fixture
def mock_trajectory():
    """Create mock trajectory data for metric tests.

    Shape: (n_checkpoints, n_trajectories, n_params)
    """
    np.random.seed(42)
    # 10 checkpoints, 5 trajectories, 50 parameters
    return np.random.randn(10, 5, 50)


@pytest.fixture
def convergent_trajectory():
    """Create trajectory data that converges.

    Trajectories start spread out and converge toward same point.
    """
    np.random.seed(42)
    n_checkpoints = 10
    n_trajectories = 5
    n_params = 20

    # Start with spread-out points
    trajectory = np.zeros((n_checkpoints, n_trajectories, n_params))

    # Initial points spread out
    for i in range(n_trajectories):
        trajectory[0, i, :] = np.random.randn(n_params) * (i + 1) * 0.1

    # Converge toward origin over time
    for t in range(1, n_checkpoints):
        decay = 0.8**t
        for i in range(n_trajectories):
            trajectory[t, i, :] = (
                trajectory[0, i, :] * decay + np.random.randn(n_params) * 0.01
            )

    return trajectory


@pytest.fixture
def divergent_trajectory():
    """Create trajectory data that diverges.

    Trajectories start close and spread apart.
    """
    np.random.seed(42)
    n_checkpoints = 10
    n_trajectories = 5
    n_params = 20

    trajectory = np.zeros((n_checkpoints, n_trajectories, n_params))

    # Start close together
    base = np.random.randn(n_params)
    for i in range(n_trajectories):
        trajectory[0, i, :] = base + np.random.randn(n_params) * 0.01

    # Diverge over time
    for t in range(1, n_checkpoints):
        growth = 1.2**t
        for i in range(n_trajectories):
            direction = np.random.randn(n_params)
            direction = direction / np.linalg.norm(direction)
            trajectory[t, i, :] = trajectory[t - 1, i, :] + direction * 0.1 * growth

    return trajectory


@pytest.fixture
def simple_dataloader():
    """Create a simple DataLoader for testing."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def tiny_dataloader():
    """Create a tiny DataLoader for fast tests."""
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.randn(20, 2)
    y = torch.randn(20, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=False)
