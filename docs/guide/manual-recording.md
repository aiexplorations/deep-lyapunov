# Manual Recording

For custom training loops where you need fine-grained control over checkpointing.

## Overview

Manual recording mode gives you full control:

1. You start recording when ready
2. You call `record_checkpoint()` at desired intervals
3. You compute metrics when done

## Basic Usage

```python
from deep_lyapunov import StabilityAnalyzer

analyzer = StabilityAnalyzer(model, n_trajectories=5)

# Start recording
analyzer.start_recording()

# Get model copies for training
models = analyzer.get_models()

# Your custom training loop
for epoch in range(10):
    for model in models:
        train_one_epoch(model)

    # Record checkpoint after each epoch
    analyzer.record_checkpoint()

# Compute metrics
results = analyzer.compute_metrics()
```

## Step-by-Step

### 1. Start Recording

```python
analyzer.start_recording()
```

This creates perturbed model copies and initializes tracking. The initial state is automatically recorded.

### 2. Get Model Copies

```python
models = analyzer.get_models()
```

Returns a list of `n_trajectories` models:

- `models[0]`: Unperturbed original (if `include_original=True`)
- `models[1:]`: Perturbed copies

### 3. Train Models

Train each model copy however you like:

```python
optimizers = [Adam(m.parameters()) for m in models]

for epoch in range(n_epochs):
    for i, (model, opt) in enumerate(zip(models, optimizers)):
        # Your training logic
        for X, y in train_loader:
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()

    # Record after each epoch
    analyzer.record_checkpoint()
```

### 4. Compute Metrics

```python
results = analyzer.compute_metrics()
```

You can optionally pass training metrics:

```python
training_metrics = [
    {"loss": [0.5, 0.4], "accuracy": [0.8, 0.85]},
    {"loss": [0.6, 0.45], "accuracy": [0.75, 0.82]},
]
results = analyzer.compute_metrics(training_metrics)
```

## Complete Example

```python
import torch
import torch.nn as nn
from deep_lyapunov import StabilityAnalyzer

# Setup
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
analyzer = StabilityAnalyzer(model, n_trajectories=3, device="cpu")

# Start recording
analyzer.start_recording()
models = analyzer.get_models()

# Create optimizers
optimizers = [torch.optim.Adam(m.parameters(), lr=0.01) for m in models]
criterion = nn.MSELoss()

# Training data
X = torch.randn(100, 10)
y = torch.randn(100, 5)

# Training loop
for epoch in range(10):
    for model, optimizer in zip(models, optimizers):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    analyzer.record_checkpoint()
    print(f"Epoch {epoch + 1} complete")

# Get results
results = analyzer.compute_metrics()
print(f"Convergence: {results.convergence_ratio:.3f}")
print(f"Behavior: {results.behavior}")
```

## When to Use Manual Mode

- Complex training loops (GANs, RL, meta-learning)
- Custom checkpoint intervals (every N batches)
- Parallel/distributed training
- Integration with existing training frameworks
- Fine-grained control over what gets recorded

## Resetting the Analyzer

```python
analyzer.reset()  # Clear all recorded data
```

Call this before starting a new analysis with the same analyzer.
