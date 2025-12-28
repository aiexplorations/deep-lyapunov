# Quick Start

This guide shows you how to analyze the training stability of a neural network in just a few lines of code.

## Basic Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from deep_lyapunov import StabilityAnalyzer

# 1. Create your model
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)

# 2. Prepare your data
X = torch.randn(500, 20)
y = torch.randn(500, 10)
train_loader = DataLoader(TensorDataset(X, y), batch_size=32)

# 3. Define a training function
def train_fn(model, loader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for _ in range(n_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    return {"loss": [0.1]}  # Return training metrics

# 4. Create analyzer and run analysis
analyzer = StabilityAnalyzer(
    model=model,
    perturbation_scale=0.01,  # 1% perturbation
    n_trajectories=5,          # Track 5 copies
)

results = analyzer.analyze(
    train_fn=train_fn,
    train_loader=train_loader,
    n_epochs=10,
)

# 5. View results
print(f"Convergence Ratio: {results.convergence_ratio:.2f}x")
print(f"Lyapunov Exponent: {results.lyapunov:.4f}")
print(f"Behavior: {results.behavior}")
```

## Understanding the Output

### Convergence Ratio

- **< 1.0**: Trajectories converge → Stable, reproducible training
- **= 1.0**: No change → Neutral stability
- **> 1.0**: Trajectories diverge → Sensitive to initialization

### Lyapunov Exponent

- **Negative**: Stable (trajectories converge exponentially)
- **Zero**: Neutral stability
- **Positive**: Chaotic (trajectories diverge exponentially)

## Visualizing Results

```python
# Plot weight trajectories in PCA space
results.plot_trajectories()

# Plot spread evolution over training
results.plot_convergence()

# Generate full report with all plots
results.save_report("my_analysis/")
```

## Next Steps

- Learn about [key concepts](concepts.md)
- Explore [automatic analysis](../guide/automatic-analysis.md)
- See [more examples](../examples/basic-usage.md)
