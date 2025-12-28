# Perturbation Analysis

How deep-lyapunov measures training stability through controlled perturbations.

## The Method

### Core Idea

1. Start with a base model initialization $\theta_0$
2. Create $N$ perturbed copies: $\theta_0^{(i)} = \theta_0 + \epsilon \cdot \eta^{(i)}$
3. Train all copies on the same data
4. Measure how the copies diverge or converge

### Perturbation Strategy

Perturbations are Gaussian, scaled by parameter standard deviation:

$$\theta_0^{(i)} = \theta_0 + \epsilon \cdot \sigma_\theta \cdot \mathcal{N}(0, I)$$

where:

- $\epsilon$ is the perturbation scale (default: 0.01)
- $\sigma_\theta$ is the standard deviation of each parameter
- $\mathcal{N}(0, I)$ is standard normal noise

### Why Scale by $\sigma_\theta$?

Parameters in different layers have different magnitudes. Scaling by standard deviation ensures:

- Equal relative perturbation across all parameters
- No layer dominates the perturbation
- Meaningful comparisons across architectures

## Trajectory Tracking

### Weight Vectors

At each checkpoint $t$, we record the flattened weight vector:

$$\theta_t = [\theta_t^{(1)}, \theta_t^{(2)}, ..., \theta_t^{(L)}]$$

where $\theta_t^{(l)}$ are the parameters of layer $l$.

### Spread Measurement

Spread is computed in PCA space for numerical stability:

1. Project all weight vectors to top $k$ PCA components
2. Compute standard deviation across trajectories:

$$\text{spread}(t) = \sqrt{\sum_{j=1}^{k} \text{Var}(z_t^{(j)})}$$

where $z_t^{(j)}$ is the $j$-th PCA component at time $t$.

## Convergence Ratio

The convergence ratio compares initial and final spreads:

$$r = \frac{\text{spread}(T)}{\text{spread}(0)}$$

### Interpretation

| Ratio | Meaning |
|:------|:--------|
| $r < 1$ | Trajectories converged (stable) |
| $r = 1$ | No change |
| $r > 1$ | Trajectories diverged (sensitive) |

## PCA Projection

### Why PCA?

1. **Dimensionality reduction**: From millions to tens of dimensions
2. **Noise filtering**: Focuses on principal directions of variation
3. **Visualization**: Enables 2D/3D trajectory plots
4. **Numerical stability**: Avoids issues with very high dimensions

### How It Works

1. Collect all weight vectors: $\{\theta_t^{(i)}\}$ for all $t, i$
2. Fit PCA on concatenated data
3. Project all trajectories to PCA space
4. Analyze in reduced space

## Effective Dimensionality

### Participation Ratio

Measures how many PCA dimensions are "active":

$$\text{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

where $\lambda_i$ are PCA eigenvalues.

### Interpretation

- $\text{PR} = 1$: All variance in one dimension
- $\text{PR} = n$: Uniform variance across $n$ dimensions

## Assumptions and Limitations

### Assumptions

1. **Same data**: All copies trained on identical data
2. **Same optimizer**: Identical optimization algorithm
3. **No stochasticity**: Deterministic training (or controlled randomness)

### Limitations

1. **Computational cost**: Training $N$ models instead of 1
2. **Memory**: Storing multiple model copies
3. **Linear approximation**: PCA assumes linear relationships
4. **Local analysis**: Results depend on starting point

## Best Practices

### Choosing Perturbation Scale

- **Too small**: Numerical precision issues
- **Too large**: Non-linear effects dominate
- **Recommended**: 0.001 to 0.1 (0.1% to 10%)

### Number of Trajectories

- **Minimum**: 3 (for meaningful statistics)
- **Recommended**: 5-10
- **More**: Diminishing returns, increased cost

### Number of PCA Components

- **Default**: 10
- **For visualization**: 2-3
- **For accurate metrics**: 5-20
