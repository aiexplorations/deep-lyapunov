# deep-lyapunov

**Analyze neural network training stability using Lyapunov exponents and perturbation-based trajectory analysis.**

[![PyPI version](https://badge.fury.io/py/deep-lyapunov.svg)](https://badge.fury.io/py/deep-lyapunov)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is deep-lyapunov?

deep-lyapunov helps you understand **how stable your neural network training is**. It answers questions like:

- Do small changes in initialization lead to similar or different final models?
- Is my training process reproducible?
- Which architectures are more stable to train?

It uses concepts from **dynamical systems theory**—specifically Lyapunov exponents—to quantify training stability.

## Key Concepts

| Metric | What It Tells You |
|:-------|:------------------|
| **Convergence Ratio** | Do weight trajectories come together (< 1) or spread apart (> 1)? |
| **Lyapunov Exponent** | Rate of trajectory separation. Negative = stable, Positive = chaotic |
| **Effective Dimensionality** | How many degrees of freedom in your weight dynamics? |

## Installation

```bash
pip install deep-lyapunov
```

For development:
```bash
pip install deep-lyapunov[dev]
```

## Quick Start

```python
import torch
import torch.nn as nn
from deep_lyapunov import StabilityAnalyzer

# Your model (any PyTorch nn.Module)
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create analyzer
analyzer = StabilityAnalyzer(
    model=model,
    perturbation_scale=0.01,  # 1% perturbation
    n_trajectories=5,          # Compare 5 perturbed copies
)

# Analyze stability during training
results = analyzer.analyze(
    train_fn=your_training_function,
    train_loader=train_loader,
    n_epochs=10,
)

# View results
print(f"Convergence Ratio: {results.convergence_ratio:.2f}x")
print(f"Lyapunov Exponent: {results.lyapunov:.4f}")
print(f"Training is: {results.behavior}")  # 'convergent' or 'divergent'

# Generate visualizations
results.plot_trajectories()
results.save_report("stability_report/")
```

## Manual Recording Mode

If you have a custom training loop:

```python
analyzer = StabilityAnalyzer(model)

analyzer.start_recording()
for epoch in range(10):
    train_one_epoch(model, train_loader)
    analyzer.record_checkpoint()

results = analyzer.compute_metrics()
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

## API Reference

### StabilityAnalyzer

```python
StabilityAnalyzer(
    model: nn.Module,
    perturbation_scale: float = 0.01,
    n_trajectories: int = 5,
    n_pca_components: int = 10,
    track_gradients: bool = False,
    device: str = "auto",
)
```

### AnalysisResults

```python
results.convergence_ratio  # float: final_spread / initial_spread
results.lyapunov           # float: average Lyapunov exponent
results.behavior           # str: 'convergent' or 'divergent'
results.spread_evolution   # np.ndarray: spread at each epoch
results.pca_trajectories   # np.ndarray: trajectories in PCA space

results.plot_trajectories()  # Visualize weight trajectories
results.plot_convergence()   # Plot spread evolution
results.save_report(path)    # Generate full report
results.to_dict()            # Export as dictionary
```

## Examples

See the [examples/](examples/) directory for:
- `basic_usage.py` - Simple stability analysis
- `custom_training.py` - Integration with custom training loops
- `comparing_architectures.py` - Compare stability across architectures

## Theory

The analysis is based on **perturbation-based trajectory analysis**:

1. Start with a base model initialization
2. Create N copies with small Gaussian perturbations (scale ε)
3. Train all copies independently on the same data
4. Track how the weight trajectories evolve
5. Compute stability metrics from trajectory spread

The **Lyapunov exponent** λ is computed as:
```
λ = (1/T) × log(final_spread / initial_spread)
```

- λ < 0: Trajectories converge (stable)
- λ > 0: Trajectories diverge (chaotic/sensitive)
- λ ≈ 0: Neutral stability

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use deep-lyapunov in your research, please cite:

```bibtex
@software{deep_lyapunov,
  title = {deep-lyapunov: Neural Network Training Stability Analysis},
  author = {Sampathkumar, Rajesh},
  year = {2025},
  url = {https://github.com/aiexplorations/deep-lyapunov}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
