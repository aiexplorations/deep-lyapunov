"""Basic usage example for deep-lyapunov.

This example demonstrates how to analyze the training stability of a simple
neural network using the StabilityAnalyzer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deep_lyapunov import StabilityAnalyzer


def create_model() -> nn.Module:
    """Create a simple MLP for demonstration."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )


def create_data(n_samples: int = 500) -> DataLoader:
    """Create synthetic classification data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, 20)
    y = torch.randn(n_samples, 10)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def train_fn(model: nn.Module, train_loader: DataLoader, n_epochs: int) -> dict:
    """Simple training function.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        n_epochs: Number of training epochs

    Returns:
        Dictionary with training metrics
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    return {"loss": losses}


def main():
    """Run basic stability analysis."""
    print("=" * 60)
    print("deep-lyapunov: Basic Usage Example")
    print("=" * 60)

    # Create model and data
    model = create_model()
    train_loader = create_data()

    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Data: {len(train_loader.dataset)} samples")

    # Create analyzer
    analyzer = StabilityAnalyzer(
        model=model,
        perturbation_scale=0.01,  # 1% perturbation
        n_trajectories=5,          # Compare 5 perturbed copies
        n_pca_components=10,
        device="cpu",
        verbose=True,
    )

    print("\n" + "-" * 60)
    print("Starting stability analysis...")
    print("-" * 60)

    # Run analysis
    results = analyzer.analyze(
        train_fn=train_fn,
        train_loader=train_loader,
        n_epochs=10,
    )

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Convergence Ratio: {results.convergence_ratio:.3f}x")
    print(f"Lyapunov Exponent: {results.lyapunov:.4f}")
    print(f"Behavior: {results.behavior.upper()}")
    print(f"Effective Dimensionality: {results.effective_dimensionality:.1f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    if results.behavior == "convergent":
        print("Training is STABLE:")
        print("  - Different initializations converge to similar solutions")
        print("  - Training is reproducible")
        print("  - Good for production deployment")
    else:
        print("Training is SENSITIVE:")
        print("  - Small differences amplify during training")
        print("  - Multiple distinct solutions exist")
        print("  - Consider using ensemble methods")

    # Generate visualizations
    print("\n" + "-" * 60)
    print("Saving visualizations...")
    print("-" * 60)
    results.save_report("stability_report/")
    print("Report saved to: stability_report/")

    return results


if __name__ == "__main__":
    main()
