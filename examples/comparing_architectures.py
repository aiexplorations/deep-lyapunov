"""Compare stability across different neural network architectures.

This example analyzes how different architectures affect training stability,
helping identify which designs are more robust to initialization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple

from deep_lyapunov import StabilityAnalyzer, AnalysisResults


def create_shallow_network() -> nn.Module:
    """Shallow network: fewer layers, more neurons per layer."""
    return nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def create_deep_network() -> nn.Module:
    """Deep network: more layers, fewer neurons per layer."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
    )


def create_residual_block(in_features: int, out_features: int) -> nn.Module:
    """Create a simple residual block."""
    class ResidualBlock(nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.linear1 = nn.Linear(in_f, out_f)
            self.linear2 = nn.Linear(out_f, out_f)
            self.relu = nn.ReLU()
            self.skip = nn.Linear(in_f, out_f) if in_f != out_f else nn.Identity()

        def forward(self, x):
            identity = self.skip(x)
            out = self.relu(self.linear1(x))
            out = self.linear2(out)
            return self.relu(out + identity)

    return ResidualBlock(in_features, out_features)


def create_residual_network() -> nn.Module:
    """Network with residual connections."""
    return nn.Sequential(
        create_residual_block(20, 64),
        create_residual_block(64, 32),
        nn.Linear(32, 10),
    )


def create_dropout_network() -> nn.Module:
    """Network with dropout regularization."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 10),
    )


def create_batchnorm_network() -> nn.Module:
    """Network with batch normalization."""
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
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
    """Simple training function."""
    model.train()
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
        losses.append(epoch_loss / len(train_loader))

    return {"loss": losses}


def analyze_architecture(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int = 10,
) -> Tuple[str, AnalysisResults]:
    """Analyze a single architecture."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {name}: {n_params:,} parameters")

    analyzer = StabilityAnalyzer(
        model=model,
        perturbation_scale=0.01,
        n_trajectories=5,
        device="cpu",
        verbose=False,
    )

    results = analyzer.analyze(
        train_fn=train_fn,
        train_loader=train_loader,
        n_epochs=n_epochs,
    )

    print(f"    Convergence: {results.convergence_ratio:.3f}x | "
          f"Lyapunov: {results.lyapunov:.4f} | "
          f"Behavior: {results.behavior}")

    return name, results


def main():
    """Compare stability across architectures."""
    print("=" * 70)
    print("deep-lyapunov: Architecture Comparison Example")
    print("=" * 70)

    # Define architectures to compare
    architectures = {
        "Shallow (2 layers)": create_shallow_network,
        "Deep (4 layers)": create_deep_network,
        "Residual": create_residual_network,
        "Dropout": create_dropout_network,
        "BatchNorm": create_batchnorm_network,
    }

    # Create training data
    train_loader = create_data()

    # Analyze each architecture
    print("\nAnalyzing architectures...")
    print("-" * 70)

    results: Dict[str, AnalysisResults] = {}
    for name, model_fn in architectures.items():
        model = model_fn()
        arch_name, arch_results = analyze_architecture(
            name, model, train_loader, n_epochs=8
        )
        results[arch_name] = arch_results

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Architecture':<20} {'Conv. Ratio':>12} {'Lyapunov':>10} {'Behavior':>12}")
    print("-" * 70)

    # Sort by convergence ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1].convergence_ratio)

    for name, r in sorted_results:
        behavior_symbol = "+" if r.behavior == "convergent" else "-"
        print(f"{name:<20} {r.convergence_ratio:>12.3f} {r.lyapunov:>10.4f} "
              f"{r.behavior:>11} {behavior_symbol}")

    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)

    most_stable = sorted_results[0]
    least_stable = sorted_results[-1]

    print(f"Most stable: {most_stable[0]} "
          f"(ratio={most_stable[1].convergence_ratio:.3f})")
    print(f"Least stable: {least_stable[0]} "
          f"(ratio={least_stable[1].convergence_ratio:.3f})")

    # Save individual reports
    print("\nSaving individual reports...")
    for name, r in results.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        r.save_report(f"architecture_comparison/{safe_name}/")

    print("Reports saved to: architecture_comparison/")

    return results


if __name__ == "__main__":
    main()
