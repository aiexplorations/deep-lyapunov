"""Custom training loop example for deep-lyapunov.

This example demonstrates how to use the manual recording mode
for integration with custom training loops.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lyapunov import StabilityAnalyzer


def create_model() -> nn.Module:
    """Create a simple MLP."""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
    )


def create_data(n_samples: int = 200) -> DataLoader:
    """Create synthetic data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, 10)
    y = torch.randn(n_samples, 5)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def main():
    """Run analysis with manual recording mode."""
    print("=" * 60)
    print("deep-lyapunov: Custom Training Loop Example")
    print("=" * 60)

    # Create model and data
    model = create_model()
    train_loader = create_data()

    # Create analyzer
    analyzer = StabilityAnalyzer(
        model=model,
        perturbation_scale=0.01,
        n_trajectories=3,
        device="cpu",
        verbose=True,
    )

    # Start manual recording
    print("\nStarting manual recording mode...")
    analyzer.start_recording()

    # Get the perturbed model copies
    models = analyzer.get_models()
    print(f"Created {len(models)} model copies")

    # Create optimizers for each model
    optimizers = [
        torch.optim.Adam(m.parameters(), lr=0.01)
        for m in models
    ]
    criterion = nn.MSELoss()

    # Custom training loop
    n_epochs = 10
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 40)

    for epoch in range(n_epochs):
        epoch_losses = []

        for X_batch, y_batch in train_loader:
            for model_copy, optimizer in zip(models, optimizers):
                optimizer.zero_grad()
                output = model_copy(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1:2d}: Avg Loss = {avg_loss:.4f}")

        # Record checkpoint after each epoch
        analyzer.record_checkpoint()

    # Compute metrics
    print("\n" + "-" * 40)
    print("Computing stability metrics...")
    results = analyzer.compute_metrics()

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Convergence Ratio: {results.convergence_ratio:.3f}x")
    print(f"Lyapunov Exponent: {results.lyapunov:.4f}")
    print(f"Behavior: {results.behavior.upper()}")

    # Per-trajectory analysis
    print("\nPer-trajectory statistics:")
    for tm in results.trajectory_metrics:
        print(f"  Trajectory {tm.trajectory_id}: "
              f"path_length={tm.path_length:.4f}, "
              f"velocity={tm.velocity_mean:.4f}")

    # Generate visualizations
    print("\nSaving report...")
    results.save_report("custom_training_report/")
    print("Done!")

    return results


if __name__ == "__main__":
    main()
