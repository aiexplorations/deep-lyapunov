# Interpreting Results

A guide to understanding what your stability analysis results mean.

## Quick Reference

| Metric | Good Sign | Warning Sign |
|:-------|:----------|:-------------|
| Convergence Ratio | < 0.5 | > 2.0 |
| Lyapunov Exponent | < -0.1 | > 0.1 |
| Behavior | convergent | divergent |

## Convergence Ratio

### What It Means

The ratio of final trajectory spread to initial spread:

```
ratio = std(final_weights) / std(initial_weights)
```

### Interpretation

| Ratio | Classification | Meaning |
|:------|:---------------|:--------|
| < 0.5 | Strongly convergent | Excellent stability, highly reproducible |
| 0.5-1.0 | Convergent | Good stability, reproducible results |
| 1.0 | Neutral | No amplification or damping |
| 1.0-2.0 | Mildly divergent | Some sensitivity to initialization |
| > 2.0 | Strongly divergent | High sensitivity, different solutions |

### What to Do

**If convergent (< 1.0):**

- Training is reproducible ✓
- Safe for production deployment
- Single model sufficient

**If divergent (> 1.0):**

- Consider ensemble methods
- May indicate optimization challenges
- Check learning rate, batch size
- Could be beneficial for diversity

## Lyapunov Exponent

### What It Means

The rate of exponential growth/decay of trajectory separation:

```
λ = (1/T) × log(final_spread / initial_spread)
```

### Interpretation

| λ Value | Meaning |
|:--------|:--------|
| λ << 0 (e.g., -0.3) | Strong contraction, highly stable |
| λ < 0 | Stable, trajectories converge |
| λ ≈ 0 | Neutral, trajectories maintain distance |
| λ > 0 | Unstable, trajectories diverge |
| λ >> 0 (e.g., 0.3) | Chaotic, highly sensitive |

### Relationship to Convergence Ratio

```
ratio = exp(λ × n_epochs)
```

Example: λ = -0.1 over 10 epochs → ratio ≈ 0.37

## Effective Dimensionality

### What It Means

How many dimensions are "active" in the weight dynamics, based on PCA participation ratio.

### Interpretation

| Value | Meaning |
|:------|:--------|
| Low (1-5) | Dynamics confined to few directions |
| Medium (5-20) | Moderate complexity |
| High (>20) | Complex, high-dimensional dynamics |

### Implications

- **Low dimensionality**: Training follows a simple path, possibly underfitting
- **High dimensionality**: Complex optimization landscape, may need regularization

## Spread Evolution

### Reading the Plot

The spread evolution plot shows trajectory spread over time:

- **Monotonically decreasing**: Stable convergence
- **Monotonically increasing**: Consistent divergence
- **Oscillating**: Phase transitions in training
- **Plateau then drop**: Learning rate schedule effects

### Phases to Look For

1. **Initial phase**: Often high spread as trajectories settle
2. **Middle phase**: Main training dynamics
3. **Final phase**: Convergence to solution(s)

## Per-Trajectory Metrics

### Path Length

Total distance traveled in weight space:

- **High path length**: Active optimization, large weight changes
- **Low path length**: Stable weights, possibly converged

### Velocity

Rate of weight change over time:

- **Decreasing velocity**: Converging
- **Constant velocity**: Steady state
- **Increasing velocity**: Instability

## Common Patterns

### Pattern 1: Healthy Convergent Training

- Ratio: 0.3-0.7
- Lyapunov: -0.1 to -0.05
- Spread: Smoothly decreasing
- Path length: Moderate, decreasing

### Pattern 2: Divergent but Stable

- Ratio: 1.5-3.0
- Lyapunov: 0.05 to 0.15
- Spread: Smoothly increasing
- Multiple distinct solutions exist

### Pattern 3: Chaotic Training

- Ratio: > 5.0
- Lyapunov: > 0.3
- Spread: Erratic, spiking
- Training needs debugging

## Actionable Insights

### To Increase Stability

1. **Lower learning rate**: Reduces oscillations
2. **Add regularization**: L2, dropout, batch norm
3. **Increase batch size**: Smoother gradients
4. **Use learning rate warmup**: Gentler start
5. **Add residual connections**: Easier optimization

### To Leverage Divergence

1. **Create ensembles**: Combine diverse models
2. **Use different initializations**: Explore solution space
3. **Study the solutions**: Understand the loss landscape
