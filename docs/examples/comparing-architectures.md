# Comparing Architectures

How to compare training stability across different neural network designs.

## Full Code

```python
--8<-- "examples/comparing_architectures.py"
```

## Sample Output

```
======================================================================
deep-lyapunov: Architecture Comparison Example
======================================================================

Analyzing architectures...
----------------------------------------------------------------------

  Shallow (2 layers): 2,826 parameters
    Convergence: 0.645x | Lyapunov: -0.0439 | Behavior: convergent

  Deep (4 layers): 3,402 parameters
    Convergence: 1.234x | Lyapunov: 0.0210 | Behavior: divergent

  Residual: 7,050 parameters
    Convergence: 0.512x | Lyapunov: -0.0669 | Behavior: convergent

  Dropout: 3,402 parameters
    Convergence: 0.789x | Lyapunov: -0.0237 | Behavior: convergent

  BatchNorm: 3,530 parameters
    Convergence: 0.423x | Lyapunov: -0.0861 | Behavior: convergent

======================================================================
SUMMARY COMPARISON
======================================================================
Architecture           Conv. Ratio   Lyapunov     Behavior
----------------------------------------------------------------------
BatchNorm                    0.423    -0.0861   convergent +
Residual                     0.512    -0.0669   convergent +
Shallow (2 layers)           0.645    -0.0439   convergent +
Dropout                      0.789    -0.0237   convergent +
Deep (4 layers)              1.234     0.0210    divergent -

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
Most stable: BatchNorm (ratio=0.423)
Least stable: Deep (4 layers) (ratio=1.234)
```

## Key Insights

### Batch Normalization

- Most stable architecture
- Normalizes activations, smooths loss landscape
- Recommended for production

### Residual Connections

- Very stable
- Skip connections ease optimization
- Good for deep networks

### Dropout

- Adds noise but still convergent
- Regularization doesn't hurt stability
- Good for generalization

### Deep Networks

- Can be less stable
- More sensitive to initialization
- Consider residual connections or batch norm
