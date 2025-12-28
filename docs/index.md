# deep-lyapunov

**Analyze neural network training stability using Lyapunov exponents and perturbation-based trajectory analysis.**

[![PyPI version](https://badge.fury.io/py/deep-lyapunov.svg)](https://badge.fury.io/py/deep-lyapunov)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/aiexplorations/deep-lyapunov/workflows/Tests/badge.svg)](https://github.com/aiexplorations/deep-lyapunov/actions)

## What is deep-lyapunov?

deep-lyapunov helps you understand **how stable your neural network training is**. It answers questions like:

- Do small changes in initialization lead to similar or different final models?
- Is my training process reproducible?
- Which architectures are more stable to train?

It uses concepts from **dynamical systems theory**—specifically Lyapunov exponents—to quantify training stability.

## Key Metrics

| Metric | What It Tells You |
|:-------|:------------------|
| **Convergence Ratio** | Do weight trajectories come together (< 1) or spread apart (> 1)? |
| **Lyapunov Exponent** | Rate of trajectory separation. Negative = stable, Positive = chaotic |
| **Effective Dimensionality** | How many degrees of freedom in your weight dynamics? |

## Quick Example

```python
import torch.nn as nn
from deep_lyapunov import StabilityAnalyzer

# Your model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Analyze stability
analyzer = StabilityAnalyzer(
    model=model,
    perturbation_scale=0.01,
    n_trajectories=5,
)

results = analyzer.analyze(
    train_fn=your_training_function,
    train_loader=train_loader,
    n_epochs=10,
)

# View results
print(f"Convergence Ratio: {results.convergence_ratio:.2f}x")
print(f"Lyapunov Exponent: {results.lyapunov:.4f}")
print(f"Training is: {results.behavior}")

# Generate visualizations
results.plot_trajectories()
results.save_report("stability_report/")
```

## Installation

```bash
pip install deep-lyapunov
```

For development:
```bash
pip install deep-lyapunov[dev]
```

## Understanding Results

### Convergent Training (ratio < 1.0)
- Different initializations converge to similar solutions
- Training is **reproducible**
- Good for production deployment

### Divergent Training (ratio > 1.0)
- Small differences amplify during training
- Multiple distinct solutions exist
- Good for **ensemble diversity**

## Next Steps

- [Quick Start Guide](getting-started/quickstart.md)
- [Key Concepts](getting-started/concepts.md)
- [API Reference](api/analyzer.md)
- [Examples](examples/basic-usage.md)
