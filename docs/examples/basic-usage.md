# Basic Usage Example

A complete example showing how to analyze training stability.

## Full Code

```python
--8<-- "examples/basic_usage.py"
```

## Running the Example

```bash
cd examples
python basic_usage.py
```

## Expected Output

```
============================================================
deep-lyapunov: Basic Usage Example
============================================================

Model: 11,466 parameters
Data: 500 samples

------------------------------------------------------------
Starting stability analysis...
------------------------------------------------------------
Starting stability analysis with 5 trajectories
  Training trajectory 1/5...
  Epoch 1: Loss = 1.0234
  ...
  Training trajectory 5/5...
  ...

============================================================
RESULTS
============================================================
Convergence Ratio: 0.834x
Lyapunov Exponent: -0.0181
Behavior: CONVERGENT
Effective Dimensionality: 4.2

------------------------------------------------------------
INTERPRETATION
------------------------------------------------------------
Training is STABLE:
  - Different initializations converge to similar solutions
  - Training is reproducible
  - Good for production deployment

------------------------------------------------------------
Saving visualizations...
------------------------------------------------------------
Report saved to: stability_report/
```

## Generated Files

After running, you'll find in `stability_report/`:

- `trajectories.png` - Weight trajectories in PCA space
- `convergence.png` - Spread evolution over training
- `lyapunov.png` - Local Lyapunov exponents
- `metrics.json` - Numerical results
- `report.md` - Markdown summary
