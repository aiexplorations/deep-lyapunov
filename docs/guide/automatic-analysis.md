# Automatic Analysis

The simplest way to analyze training stability using the `analyze()` method.

## Overview

Automatic analysis handles everything for you:

1. Creates perturbed model copies
2. Trains each copy using your training function
3. Records weight trajectories
4. Computes stability metrics

## Basic Usage

```python
from deep_lyapunov import StabilityAnalyzer

analyzer = StabilityAnalyzer(
    model=model,
    perturbation_scale=0.01,
    n_trajectories=5,
)

results = analyzer.analyze(
    train_fn=your_train_function,
    train_loader=train_loader,
    n_epochs=10,
)
```

## Training Function Requirements

Your training function must have this signature:

```python
def train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    **kwargs
) -> dict:
    """Train the model.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        n_epochs: Number of epochs to train

    Returns:
        Dictionary with training metrics, e.g., {'loss': [0.5, 0.4, 0.3]}
    """
    # Your training code here
    return {"loss": losses}
```

!!! note
    The function is called once per epoch with `n_epochs=1`, allowing the analyzer to record checkpoints between epochs.

## Example Training Function

```python
def train_fn(model, train_loader, n_epochs, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(train_loader))

    return {"loss": losses, "accuracy": [0.95]}  # Optional: include accuracy
```

## Configuration Options

```python
analyzer = StabilityAnalyzer(
    model=model,
    perturbation_scale=0.01,  # 1% of parameter std
    n_trajectories=5,          # Number of perturbed copies
    n_pca_components=10,       # PCA dimensions for analysis
    track_gradients=False,     # Also track gradient statistics
    device="auto",             # "cpu", "cuda", "mps", or "auto"
    seed=42,                   # For reproducibility
    verbose=True,              # Print progress
)
```

## Accessing Results

```python
results = analyzer.analyze(...)

# Core metrics
print(results.convergence_ratio)  # Final/initial spread ratio
print(results.lyapunov)           # Lyapunov exponent
print(results.behavior)           # "convergent" or "divergent"

# Detailed data
print(results.spread_evolution)   # Spread at each checkpoint
print(results.pca_trajectories)   # Projected trajectories
print(results.effective_dimensionality)

# Per-trajectory info
for tm in results.trajectory_metrics:
    print(f"Trajectory {tm.trajectory_id}: loss={tm.final_loss}")
```

## Limitations

The automatic mode assumes:

- Training function is deterministic given the model state
- Each call to `train_fn` with `n_epochs=1` trains for exactly one epoch
- The model architecture stays constant during training
